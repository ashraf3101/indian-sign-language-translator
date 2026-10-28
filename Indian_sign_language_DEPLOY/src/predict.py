"""
predict.py
Predicts Indian Sign Language (ISL) gestures in real-time from webcam
using the trained CNN model (models/isl_model.pkl).
"""

import os
import time
import pickle
import collections
import cv2
import numpy as np

try:
    from src.preprocess import extract_landmarks, get_hands_detector, TOTAL_LANDMARK_FEATURES
except ImportError:
    try:
        from preprocess import extract_landmarks, get_hands_detector, TOTAL_LANDMARK_FEATURES
    except ImportError:
        extract_landmarks = None
        get_hands_detector = None
        TOTAL_LANDMARK_FEATURES = 126


def load_model_and_encoder(model_path="models/isl_model.pkl", encoder_path="models/label_encoder.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


def run_realtime_prediction(camera_id=0, min_confidence=0.60):
    try:
        model, label_encoder = load_model_and_encoder()
        classes = label_encoder.classes_
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return

    detector = get_hands_detector(static_image_mode=False) if get_hands_detector else None
    prediction_history = collections.deque(maxlen=7)
    formed_sentence = ""
    last_added_sign = None
    consistent_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()

        predicted_label = "No Hand Detected"
        confidence = 0.0

        if extract_landmarks:
            features, display_frame, hands_count = extract_landmarks(display_frame, detector=detector, draw_landmarks=True)
            if hands_count > 0:
                probs = model.predict_proba(features.reshape(1, -1))[0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])

                if confidence >= min_confidence:
                    raw_pred = classes[top_idx]
                    prediction_history.append(raw_pred)
                    if len(prediction_history) == 7:
                        counts = collections.Counter(prediction_history)
                        smooth_pred, count = counts.most_common(1)[0]
                        if count >= 4:
                            predicted_label = smooth_pred
                            if predicted_label == last_added_sign:
                                consistent_frames += 1
                                if consistent_frames == 14:
                                    formed_sentence += (" " if len(predicted_label) > 1 else "") + predicted_label
                            else:
                                last_added_sign = predicted_label
                                consistent_frames = 1
            else:
                prediction_history.clear()
                last_added_sign = None
                consistent_frames = 0

        cv2.rectangle(display_frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(display_frame, f"Sign: {predicted_label} ({confidence*100:.0f}%)", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.rectangle(display_frame, (0, h - 60), (w, h), (15, 15, 15), -1)
        cv2.putText(display_frame, f"Sentence: {formed_sentence}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ISL Real-Time Predictor", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            formed_sentence += " "
        elif key == ord('c'):
            formed_sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    if detector:
        detector.close()


if __name__ == "__main__":
    run_realtime_prediction()
