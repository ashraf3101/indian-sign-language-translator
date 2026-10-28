"""
text_to_sign.py
Translates English text into Indian Sign Language (ISL) visual signs.
"""

import os
import re
import argparse
from PIL import Image, ImageDraw


def sanitize_token(token):
    return re.sub(r'[^A-Z0-9]', '', token.upper())


def get_sign_asset_path(token, assets_dir="assets/signs"):
    token = token.upper().replace(" ", "_")
    for ext in [".png", ".jpg", ".jpeg", ".gif"]:
        path = os.path.join(assets_dir, f"{token}{ext}")
        if os.path.exists(path):
            return path
    return None


def text_to_sign_sequence(text, assets_dir="assets/signs"):
    if not text:
        return []

    words = text.strip().upper().split()
    sequence = []

    for w_idx, word in enumerate(words):
        clean_word = sanitize_token(word)
        if not clean_word:
            continue

        word_asset = get_sign_asset_path(clean_word, assets_dir)
        if word_asset:
            sequence.append({"token": clean_word, "type": "word", "image_path": word_asset})
        else:
            for char in clean_word:
                char_asset = get_sign_asset_path(char, assets_dir)
                sequence.append({"token": char, "type": "char", "image_path": char_asset})

        if w_idx < len(words) - 1:
            sequence.append({"token": "SPACE", "type": "space", "image_path": get_sign_asset_path("SPACE", assets_dir)})

    return sequence


def create_sign_collage(sequence, img_size=(160, 160), max_cols=6):
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
            draw.rectangle([x, y, x + w, y + h], fill=(220, 225, 230), outline=(180, 185, 190), width=2)
            draw.text((x + w // 2 - 10, y + h // 2 - 10), item["token"], fill=(50, 50, 50))

        caption = item["token"]
        draw.text((x + w // 2 - len(caption) * 4, y + h + 8), caption, fill=(30, 30, 30))

    return collage


def main():
    parser = argparse.ArgumentParser(description="Translate text to ISL signs.")
    parser.add_argument("--text", type=str, default="HELLO ISL")
    args = parser.parse_args()
    seq = text_to_sign_sequence(args.text)
    for s in seq:
        print(f"[{s['type']}] {s['token']}: {s['image_path']}")


if __name__ == "__main__":
    main()
