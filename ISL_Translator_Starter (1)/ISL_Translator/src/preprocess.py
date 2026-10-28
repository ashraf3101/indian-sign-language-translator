"""
Optional dataset inspection helper.

The CNN training pipeline reads image folders directly, so it does not need a separate
landmark-extraction preprocessing stage. Run this script to verify split directories
and class counts before training.
"""
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {
    "train": {"training (a-z)", "training", "train"},
    "validation": {"validation (a-z)", "validation", "valid", "val"},
    "test": {"testing (a-z)", "testing", "test"},
}

def main():
    if not RAW_DIR.exists():
        print(f"Dataset folder not found: {RAW_DIR}")
        return
    found = {}
    for path in RAW_DIR.rglob("*"):
        if path.is_dir():
            for split, names in SPLIT_NAMES.items():
                if path.name.casefold() in names:
                    found[split] = path
    if not found:
        print("No train/validation/test split folders found. Check README.md.")
        return
    for split, folder in found.items():
        counts = {}
        for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            counts[class_dir.name] = sum(
                1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        print(f"\n{split.upper()}: {folder}")
        for label, count in counts.items():
            print(f"  {label}: {count} images")
        print(f"  classes: {len(counts)}, total images: {sum(counts.values())}")

if __name__ == "__main__":
    main()
