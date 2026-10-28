"""
text_to_sign.py
Translates English text into Indian Sign Language (ISL) visual signs.
Looks for word-level gesture assets first, then falls back to finger-spelling (A-Z, 0-9).
"""

import os
import re
import argparse
from PIL import Image, ImageDraw, ImageFont


def sanitize_token(token):
    """Sanitizes token to uppercase alphanumeric string."""
    return re.sub(r'[^A-Z0-9]', '', token.upper())


def get_sign_asset_path(token, assets_dir="assets/signs"):
    """
    Finds the image or GIF asset path for a sign token.
    Checks exact matches: TOKEN.png, TOKEN.jpg, TOKEN.gif
    """
    token = token.upper().replace(" ", "_")
    for ext in [".png", ".jpg", ".jpeg", ".gif"]:
        path = os.path.join(assets_dir, f"{token}{ext}")
        if os.path.exists(path):
            return path
    return None


def text_to_sign_sequence(text, assets_dir="assets/signs"):
    """
    Translates input text into a sequence of ISL sign items.
    Each item is a dictionary with:
      - token: str (e.g., 'HELLO', 'A', 'SPACE')
      - type: 'word', 'char', or 'space'
      - image_path: path to image asset or None
    """
    if not text:
        return []

    words = text.strip().upper().split()
    sequence = []

    for w_idx, word in enumerate(words):
        clean_word = sanitize_token(word)
        if not clean_word:
            continue

        # 1. Check if whole-word gesture asset exists (e.g., HELLO, THANK_YOU)
        word_asset = get_sign_asset_path(clean_word, assets_dir)
        if word_asset:
            sequence.append({
                "token": clean_word,
                "type": "word",
                "image_path": word_asset
            })
        else:
            # 2. Finger-spell letter by letter
            for char in clean_word:
                char_asset = get_sign_asset_path(char, assets_dir)
                sequence.append({
                    "token": char,
                    "type": "char",
                    "image_path": char_asset
                })

        # Add space between words if not the last word
        if w_idx < len(words) - 1:
            sequence.append({
                "token": "SPACE",
                "type": "space",
                "image_path": get_sign_asset_path("SPACE", assets_dir)
            })

    return sequence


def create_sign_collage(sequence, img_size=(160, 160), max_cols=6):
    """
    Stitches sign images in sequence into a single collage image.
    Useful for quick display or sharing.
    """
    valid_items = [item for item in sequence if item["type"] != "space" and item["image_path"]]
    if not valid_items:
        return None

    num_items = len(valid_items)
    cols = min(num_items, max_cols)
    rows = (num_items + cols - 1) // cols

    w, h = img_size
    margin = 15
    collage_w = cols * w + (cols + 1) * margin
    collage_h = rows * (h + 30) + (rows + 1) * margin

    collage = Image.new("RGB", (collage_w, collage_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(collage)

    for i, item in enumerate(valid_items):
        r = i // cols
        c = i % cols
        x = margin + c * (w + margin)
        y = margin + r * (h + 30 + margin)

        if item["image_path"] and os.path.exists(item["image_path"]):
            sign_img = Image.open(item["image_path"]).convert("RGB").resize(img_size)
            collage.paste(sign_img, (x, y))
        else:
            # Draw placeholder card
            draw.rectangle([x, y, x + w, y + h], fill=(220, 225, 230), outline=(180, 185, 190), width=2)
            draw.text((x + w // 2 - 10, y + h // 2 - 10), item["token"], fill=(50, 50, 50))

        # Caption text
        caption = item["token"]
        draw.text((x + w // 2 - len(caption) * 4, y + h + 8), caption, fill=(30, 30, 30))

    return collage


def main():
    parser = argparse.ArgumentParser(description="Translate text to Indian Sign Language (ISL).")
    parser.add_argument("--text", type=str, default="HELLO ISL", help="Text to translate into ISL")
    parser.add_argument("--save", type=str, default="assets/translation_output.png", help="Path to save collage")
    args = parser.parse_args()

    seq = text_to_sign_sequence(args.text)
    print(f"[*] Translation sequence for '{args.text}':")
    for s in seq:
        print(f"    - [{s['type'].upper()}] {s['token']}: {s['image_path']}")

    collage = create_sign_collage(seq)
    if collage:
        collage.save(args.save)
        print(f"[+] Saved sign sequence collage to '{args.save}'")
    else:
        print("[!] No visual assets found to create collage.")


if __name__ == "__main__":
    main()
