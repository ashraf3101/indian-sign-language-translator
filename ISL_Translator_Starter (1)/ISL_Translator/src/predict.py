import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)

def load_artifacts(model_path, labels_path):
    model = load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    # labels.json stores a list where each index corresponds to model output index.
    return model, labels

def predict_image(image: Image.Image, model, labels):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    # MobileNetV2 preprocessing must match training.
    batch = batch / 127.5 - 1.0
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1][:3]
    best_idx = int(order[0])
    top_predictions = [(labels[int(i)], float(probabilities[int(i)])) for i in order]
    return labels[best_idx], float(probabilities[best_idx]), top_predictions
