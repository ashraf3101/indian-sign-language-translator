"""
preprocess.py
Extracts and normalizes hand landmarks for Indian Sign Language (ISL) recognition.
Supports up to 2 hands (crucial for ISL two-handed signs like B, D, P, Q, etc.).
"""

import os
import cv2
import numpy as np

# Safe import for mediapipe in headless / cloud environments
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    mp = None

# Constants
MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3  # (x, y, z)
TOTAL_LANDMARK_FEATURES = MAX_HANDS * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 126

# Initialize MediaPipe Hands if available
if MP_AVAILABLE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
else:
    mp_hands = None
    mp_drawing = None
    mp_drawing_styles = None


def get_hands_detector(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Returns an initialized MediaPipe Hands instance.
    """
    if not MP_AVAILABLE or mp_hands is None:
        return None
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )


def normalize_hand_landmarks(landmarks_list):
    """
    Normalizes a list of (x, y, z) hand landmarks:
    1. Translates wrist (index 0) to origin (0, 0, 0).
    2. Scales by maximum distance from wrist to maintain scale invariance.
    """
    if not landmarks_list or len(landmarks_list) == 0:
        return np.zeros((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
    wrist = coords[0]
    
    # 1. Translate wrist to (0, 0, 0)
    translated = coords - wrist

    # 2. Scale normalize
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)
    if max_dist > 1e-6:
        normalized = translated / max_dist
    else:
        normalized = translated

    return normalized


def extract_landmarks(frame, detector=None, draw_landmarks=False):
    """
    Extracts 126-dimensional feature vector from an input RGB/BGR image frame.
    Supports up to 2 hands (42 landmarks * 3 coordinates).
    
    Parameters:
        frame: BGR numpy image array.
        detector: mp_hands.Hands instance (created if None).
        draw_landmarks: bool, whether to draw hand skeletal joints onto frame.
        
    Returns:
        tuple: (feature_vector of shape (126,), annotated_frame, hands_detected_count)
    """
    if frame is None:
        return np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32), None, 0

    annotated_frame = frame.copy() if draw_landmarks else frame
    
    # Check mediapipe availability
    if not MP_AVAILABLE or mp_hands is None:
        # Fallback dummy features if mediapipe is not installed
        return np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32), annotated_frame, 0

    created_detector = False
    if detector is None:
        detector = get_hands_detector(static_image_mode=True)
        created_detector = True

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    feature_vector = np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32)
    hands_count = 0

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            # Normalize hand coordinates
            norm_hand = normalize_hand_landmarks(hand_landmarks.landmark)
            start_idx = i * (LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK)
            end_idx = start_idx + (LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK)
            feature_vector[start_idx:end_idx] = norm_hand.flatten()

            # Draw visual landmarks if requested
            if draw_landmarks and mp_drawing:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

    if created_detector:
        detector.close()

    return feature_vector, annotated_frame, hands_count


def extract_landmarks_from_image_path(image_path, detector=None):
    """
    Loads an image from file and extracts normalized landmarks.
    """
    if not os.path.exists(image_path):
        return None, 0
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    features, _, hands_count = extract_landmarks(img, detector=detector, draw_landmarks=False)
    return features, hands_count


def process_dataset(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Processes all raw gesture images from raw_dir and saves processed feature arrays.
    Expected folder structure:
    data/raw/
        A/
            img_001.jpg
        B/
            img_001.jpg
    Saves:
        data/processed/X.npy
        data/processed/y.npy
        data/processed/classes.npy
    """
    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        print(f"[!] Raw data directory '{raw_dir}' does not exist.")
        return None, None, None

    classes = [d for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))]
    if not classes:
        print(f"[!] No class subdirectories found in '{raw_dir}'.")
        return None, None, None

    print(f"[*] Found {len(classes)} classes: {classes}")
    
    detector = get_hands_detector(static_image_mode=True)
    X = []
    y = []

    for label_idx, class_name in enumerate(classes):
        class_folder = os.path.join(raw_dir, class_name)
        img_names = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[*] Processing class '{class_name}': {len(img_names)} images...")
        
        valid_samples = 0
        for img_name in img_names:
            img_path = os.path.join(class_folder, img_name)
            features, hands_count = extract_landmarks_from_image_path(img_path, detector)
            if features is not None and hands_count > 0:
                X.append(features)
                y.append(class_name)
                valid_samples += 1
                
        print(f"    - Extracted {valid_samples} valid hand samples for '{class_name}'.")

    if detector:
        detector.close()

    if len(X) == 0:
        print("[!] No landmarks could be extracted from images.")
        return None, None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    np.save(os.path.join(processed_dir, "X.npy"), X)
    np.save(os.path.join(processed_dir, "y.npy"), y)
    np.save(os.path.join(processed_dir, "classes.npy"), np.array(classes))

    print(f"[+] Dataset processed successfully! Shapes: X={X.shape}, y={y.shape}")
    return X, y, classes


if __name__ == "__main__":
    print("[*] Running preprocess.py...")
    process_dataset()
