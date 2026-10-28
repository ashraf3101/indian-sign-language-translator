# Models Directory

This directory contains the trained Convolutional Neural Network (CNN) model and label encoder for Indian Sign Language (ISL) recognition:

- `isl_model.pkl`: Serialized deep 1D-CNN classifier model trained on normalized 126-dimensional MediaPipe hand joint features (both hands).
- `label_encoder.pkl`: Scikit-learn `LabelEncoder` instance mapping 40 ISL gesture classes to integer indices.

## Training
To retrain or update the model:
```bash
python src/train_model.py
```
This generates an updated `isl_model.pkl` and `label_encoder.pkl` along with training convergence metrics in `assets/training_metrics.png`.
