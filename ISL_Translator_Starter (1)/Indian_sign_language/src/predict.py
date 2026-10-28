"""
predict.py
Predicts Indian Sign Language (ISL) gestures in real-time from webcam
using the trained CNN model (models/isl_model.pkl).
Builds words and sentences with speech output and smoothing filter.
"""

import os
import sys
import time
import pickle
import collections
import cv2
import numpy as np

# Import preprocess module
try:
    from src.preprocess import extract_landmarks, get_hands_detector, TOTAL_LANDMARK_FEATURES, MP_AVAILABLE
except ImportError:
    try:
        from preprocess import extract_landmarks, get_hands_detector, TOTAL_LANDMARK_FEATURES, MP_AVAILABLE
    except ImportError:
        extract_landmarks = None
        get_hands_detector = None
        TOTAL_LANDMARK_FEATURES = 126
        MP_AVAILABLE = False


def load_model_and_encoder(model_path="models/isl_model.pkl", encoder_path="models/label_encoder.pkl"):
    """
    Loads trained CNN model and label encoder from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run src/train_model.py first.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file '{encoder_path}' not found. Please run src/train_model.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


def speak_text(text):
    """
    Speaks text aloud using pyttsx3 or terminal announcement.
    """
    if not text.strip():
        return
    print(f"\n[SPEECH]: \"{text}\"")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        # Graceful fallback if audio device or pyttsx3 is unavailable
        pass


def run_realtime_prediction(camera_id=0, min_confidence=0.60):
    """
    Runs real-time sign language prediction loop.
    """
    print("[*] Loading ISL CNN model and label encoder...")
    try:
        model, label_encoder = load_model_and_encoder()
        classes = label_encoder.classes_
        print(f"[+] Model loaded successfully with {len(classes)} classes: {list(classes[:10])}...")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[!] Error: Could not access camera {camera_id}.")
        return

    detector = get_hands_detector(static_image_mode=False) if get_hands_detector else None
    
    # Smoothing buffer
    SMOOTHING_WINDOW = 7
    prediction_history = collections.deque(maxlen=SMOOTHING_WINDOW)
    
    # Sentence formation variables
    formed_sentence = ""
    last_added_sign = None
    consistent_frames = 0
    FRAMES_REQUIRED_FOR_COMMIT = 14  # ~0.5 sec of holding sign steady

    prev_frame_time = time.time()

    print("\n" + "=" * 60)
    print("REAL-TIME INDIAN SIGN LANGUAGE PREDICTOR")
    print("Controls:")
    print("  - Hold sign steady to commit character/word")
    print("  - Press 'SPACE': Insert space into sentence")
    print("  - Press 'BACKSPACE' / 'd': Delete last character")
    print("  - Press 'c': Clear current sentence")
    print("  - Press 's': Speak sentence aloud")
    print("  - Press 'q': Exit")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Unable to read video frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_frame_time + 1e-6)
        prev_frame_time = curr_time

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if extract_landmarks:
            features, display_frame, hands_count = extract_landmarks(
                display_frame, detector=detector, draw_landmarks=True
            )

            if hands_count > 0:
                features_batch = features.reshape(1, -1)
                probs = model.predict_proba(features_batch)[0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])

                if confidence >= min_confidence:
                    raw_pred = classes[top_idx]
                    prediction_history.append(raw_pred)

                    # Majority voting over smoothing window
                    if len(prediction_history) == SMOOTHING_WINDOW:
                        counts = collections.Counter(prediction_history)
                        smooth_pred, count = counts.most_common(1)[0]
                        if count >= (SMOOTHING_WINDOW // 2 + 1):
                            predicted_label = smooth_pred

                            # Sentence commitment logic
                            if predicted_label == last_added_sign:
                                consistent_frames += 1
                                if consistent_frames == FRAMES_REQUIRED_FOR_COMMIT:
                                    if len(predicted_label) == 1:
                                        formed_sentence += predicted_label
                                    else:
                                        # Whole word sign like "HELLO", "YES"
                                        formed_sentence += (" " if formed_sentence and not formed_sentence.endswith(" ") else "") + predicted_label + " "
                                    print(f"[*] Committed: '{predicted_label}' -> Current: \"{formed_sentence}\"")
                            else:
                                last_added_sign = predicted_label
                                consistent_frames = 1
                else:
                    predicted_label = "Uncertain"
            else:
                prediction_history.clear()
                last_added_sign = None
                consistent_frames = 0

        # Draw HUD interface
        # Top banner for prediction
        cv2.rectangle(display_frame, (0, 0), (w, 75), (20, 20, 20), -1)
        pred_color = (0, 255, 0) if confidence >= 0.70 else (0, 255, 255)
        cv2.putText(display_frame, f"Sign: {predicted_label}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
        cv2.putText(display_frame, f"Conf: {confidence * 100:.1f}% | FPS: {fps:.0f}", (w - 280, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Bottom banner for formed sentence
        cv2.rectangle(display_frame, (0, h - 85), (w, h), (15, 15, 15), -1)
        cv2.putText(display_frame, "Sentence:", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        sentence_display = formed_sentence if formed_sentence else "(Show signs to construct sentence...)"
        text_color = (255, 255, 255) if formed_sentence else (120, 120, 120)
        cv2.putText(display_frame, sentence_display, (130, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_color, 2)
        
        # Sub-bar hints
        cv2.putText(display_frame, "[Space] Space | [d] Delete | [c] Clear | [s] Speak | [q] Quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Indian Sign Language CNN Real-time Recognizer", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACE
            formed_sentence += " "
        elif key == ord('d') or key == 8:  # 'd' or Backspace
            formed_sentence = formed_sentence[:-1]
        elif key == ord('c'):  # Clear
            formed_sentence = ""
        elif key == ord('s'):  # Speak
            speak_text(formed_sentence)

    cap.release()
    cv2.destroyAllWindows()
    if detector:
        detector.close()


if __name__ == "__main__":
    run_realtime_prediction()
