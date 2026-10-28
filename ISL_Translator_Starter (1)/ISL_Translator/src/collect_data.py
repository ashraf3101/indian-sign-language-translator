"""
Optional webcam data collector for adding your own examples.

Usage:
    python src/collect_data.py

Creates data/own_samples/<LABEL>/ images. Do not mix these images into test data
if you want an unbiased evaluation. To use them, manually merge them into a training
class folder only after checking label conventions and the dataset license.
"""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "own_samples"

def main():
    label = input("Enter one alphabet label (A-Z): ").strip().upper()
    if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        print("Please enter one letter A-Z.")
        return

    save_dir = OUT_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press SPACE to save an image; press Q to quit.")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from webcam.")
                break
            cv2.putText(frame, f"Label: {label} | Saved: {count}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ISL data collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                target = save_dir / f"{label}_{count:04d}.jpg"
                cv2.imwrite(str(target), frame)
                count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} images to {save_dir}")

if __name__ == "__main__":
    main()
