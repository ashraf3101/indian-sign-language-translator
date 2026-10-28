# Processed ISL Dataset Directory

This directory stores extracted 126-dimensional MediaPipe hand landmarks and corresponding labels for training the CNN classifier.

## Files
- `X.npy`: Numpy array of shape `(N, 126)` containing normalized coordinates $(x, y, z)$ for Hand 1 and Hand 2.
- `y.npy`: Numpy array of shape `(N,)` containing class label strings.
- `classes.npy`: Numpy array containing unique class names.

## Generation
Run the preprocessing script to extract landmarks from `data/raw/`:

```bash
python src/preprocess.py
```

If no raw images are provided yet, `src/train_model.py` automatically synthesizes initial benchmark kinematics to train the baseline CNN model.
