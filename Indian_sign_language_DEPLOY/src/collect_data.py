"""
collect_data.py
Collects webcam image samples for Indian Sign Language (ISL) gestures.
Saves raw images into: data/raw/<label>/
"""

import os
import time
import cv2
import argparse

try:
    from src.preprocess import extract_landmarks, get_hands_detector
except ImportError:
    try:
        from preprocess import extract_landmarks, get_hands_detector
    except ImportError:
        extract_landmarks = None
        get_hands_detector = None


def collect_samples(label, num_samples=100, camera_id=0):
    label = str(label).strip().upper()
    save_dir = os.path.join("data", "raw", label)
    os.makedirs(save_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    start_index = len(existing_files)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {camera_id}.")
        return

    detector = get_hands_detector(static_image_mode=False) if get_hands_detector else None
    state = "READY"
    countdown_start = 0
    countdown_duration = 3
    captured = 0
    last_capture_time = 0
    capture_interval = 0.08

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()

        hands_count = 0
        if extract_landmarks:
            _, display_frame, hands_count = extract_landmarks(display_frame, detector=detector, draw_landmarks=True)

        current_time = time.time()

        if state == "COUNTDOWN":
            elapsed = current_time - countdown_start
            remaining = int(countdown_duration - elapsed) + 1
            if remaining > 0:
                cv2.putText(display_frame, f"Starting in {remaining}...", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)
            else:
                state = "CAPTURING"

        elif state == "CAPTURING":
            if current_time - last_capture_time >= capture_interval:
                img_name = f"{label}_{start_index + captured:04d}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, frame)
                captured += 1
                last_capture_time = current_time

            cv2.putText(display_frame, f"RECORDING: {captured}/{num_samples}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if captured >= num_samples:
                break

        cv2.rectangle(display_frame, (10, 10), (w - 10, 60), (30, 30, 30), -1)
        status_text = f"Class: {label} | Hands: {hands_count} | Mode: {state}"
        cv2.putText(display_frame, status_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 's': Start | 'p': Pause | 'q': Quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow(f"ISL Data Collector - Class {label}", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if state != "CAPTURING":
                state = "COUNTDOWN"
                countdown_start = time.time()
                captured = 0
        elif key == ord('p'):
            state = "PAUSED"

    cap.release()
    cv2.destroyAllWindows()
    if detector:
        detector.close()


def main():
    parser = argparse.ArgumentParser(description="Collect ISL gesture data via webcam.")
    parser.add_argument("--label", type=str, default="A", help="Sign class label")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()
    collect_samples(label=args.label, num_samples=args.samples, camera_id=args.camera)


if __name__ == "__main__":
    main()
