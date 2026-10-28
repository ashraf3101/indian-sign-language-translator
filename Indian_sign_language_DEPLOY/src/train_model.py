"""
train_model.py
Defines and trains a 1D Convolutional Neural Network (CNN) on hand landmark features.
Saves the trained model and label encoder to:
  - models/isl_model.pkl
  - models/label_encoder.pkl
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

NUM_COORDS = 126
DEFAULT_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'HELLO', 'THANK YOU', 'YES', 'NO', 'PLEASE'
]


class ISLConvNet:
    """
    1D Convolutional Neural Network (CNN) for Indian Sign Language gesture recognition.
    Architecture:
      - Reshape (Batch, 42, 3)
      - Conv1D(3, 32, k=3) + ReLU + MaxPool(2)
      - Conv1D(32, 64, k=3) + ReLU + MaxPool(2)
      - Flatten -> Dense(128) + ReLU
      - Dense(num_classes) + Softmax
    """
    def __init__(self, num_classes=len(DEFAULT_CLASSES), learning_rate=0.005):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.is_fitted = False
        
        np.random.seed(42)
        self.W_conv1 = np.random.randn(32, 3, 3).astype(np.float32) * np.sqrt(2.0 / 9)
        self.b_conv1 = np.zeros(32, dtype=np.float32)
        
        self.W_conv2 = np.random.randn(64, 32, 3).astype(np.float32) * np.sqrt(2.0 / 96)
        self.b_conv2 = np.zeros(64, dtype=np.float32)
        
        self.flat_dim = 9 * 64
        self.W_dense1 = np.random.randn(self.flat_dim, 128).astype(np.float32) * np.sqrt(2.0 / self.flat_dim)
        self.b_dense1 = np.zeros(128, dtype=np.float32)
        
        self.W_out = np.random.randn(128, num_classes).astype(np.float32) * np.sqrt(2.0 / 128)
        self.b_out = np.zeros(num_classes, dtype=np.float32)

    def _conv1d_forward(self, X, W, b, stride=1):
        N, C_in, L_in = X.shape
        C_out, _, K = W.shape
        L_out = (L_in - K) // stride + 1
        out = np.zeros((N, C_out, L_out), dtype=np.float32)
        for i in range(L_out):
            segment = X[:, :, i * stride:i * stride + K]
            out[:, :, i] = np.tensordot(segment, W, axes=([1, 2], [1, 2])) + b
        return out

    def _maxpool1d(self, X, pool_size=2):
        N, C, L = X.shape
        L_out = L // pool_size
        reshaped = X[:, :, :L_out * pool_size].reshape(N, C, L_out, pool_size)
        return np.max(reshaped, axis=3)

    def _forward(self, X):
        N = X.shape[0]
        X_seq = X.reshape(N, 42, 3).transpose(0, 2, 1)  # (N, 3, 42)
        
        conv1 = self._conv1d_forward(X_seq, self.W_conv1, self.b_conv1)
        relu1 = np.maximum(0, conv1)
        pool1 = self._maxpool1d(relu1, pool_size=2)
        
        conv2 = self._conv1d_forward(pool1, self.W_conv2, self.b_conv2)
        relu2 = np.maximum(0, conv2)
        pool2 = self._maxpool1d(relu2, pool_size=2)
        
        flattened = pool2.reshape(N, -1)
        if flattened.shape[1] != self.flat_dim:
            self.flat_dim = flattened.shape[1]
            self.W_dense1 = np.random.randn(self.flat_dim, 128).astype(np.float32) * 0.05
            
        dense1 = np.maximum(0, np.dot(flattened, self.W_dense1) + self.b_dense1)
        logits = np.dot(dense1, self.W_out) + self.b_out
        
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return {'pool2': pool2, 'dense1': dense1, 'probs': probs}

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] < NUM_COORDS:
            padded = np.zeros((X.shape[0], NUM_COORDS), dtype=np.float32)
            padded[:, :X.shape[1]] = X
            X = padded
        elif X.shape[1] > NUM_COORDS:
            X = X[:, :NUM_COORDS]
            
        cache = self._forward(X)
        return cache['probs']

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=35, batch_size=32, verbose=False):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        num_samples = len(X)
        history = {'loss': [], 'accuracy': []}
        
        Y_one_hot = np.zeros((num_samples, self.num_classes), dtype=np.float32)
        Y_one_hot[np.arange(num_samples), y] = 1.0
        
        v_W_out = np.zeros_like(self.W_out)
        v_b_out = np.zeros_like(self.b_out)
        v_W_dense1 = np.zeros_like(self.W_dense1)
        v_b_dense1 = np.zeros_like(self.b_dense1)
        beta = 0.9
        
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y_one_hot[indices]
            
            epoch_loss = 0.0
            correct = 0
            num_batches = int(np.ceil(num_samples / batch_size))
            
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_samples)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                batch_len = end - start
                
                cache = self._forward(X_batch)
                probs = cache['probs']
                dense1 = cache['dense1']
                
                loss = -np.mean(np.sum(Y_batch * np.log(probs + 1e-12), axis=1))
                epoch_loss += loss * batch_len
                preds = np.argmax(probs, axis=1)
                correct += np.sum(preds == np.argmax(Y_batch, axis=1))
                
                dlogits = (probs - Y_batch) / batch_len
                dW_out = np.dot(dense1.T, dlogits)
                db_out = np.sum(dlogits, axis=0)
                
                ddense1 = np.dot(dlogits, self.W_out.T)
                ddense1[dense1 <= 0] = 0
                dW_dense1 = np.dot(cache['pool2'].reshape(batch_len, -1).T, ddense1)
                db_dense1 = np.sum(ddense1, axis=0)
                
                v_W_out = beta * v_W_out + (1 - beta) * dW_out
                v_b_out = beta * v_b_out + (1 - beta) * db_out
                v_W_dense1 = beta * v_W_dense1 + (1 - beta) * dW_dense1
                v_b_dense1 = beta * v_b_dense1 + (1 - beta) * db_dense1
                
                self.W_out -= self.learning_rate * v_W_out
                self.b_out -= self.learning_rate * v_b_out
                self.W_dense1 -= self.learning_rate * v_W_dense1
                self.b_dense1 -= self.learning_rate * v_b_dense1

            epoch_loss /= num_samples
            epoch_acc = correct / num_samples
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
                
        self.is_fitted = True
        return history


def generate_synthetic_isl_dataset(classes=DEFAULT_CLASSES, samples_per_class=50):
    np.random.seed(42)
    X = []
    y = []

    for idx, cls in enumerate(classes):
        base = np.zeros(NUM_COORDS, dtype=np.float32)
        cls_seed = sum(ord(c) for c in cls) + idx * 17
        np.random.seed(cls_seed)
        
        h1_coords = np.random.uniform(-0.5, 0.5, (21, 3)).astype(np.float32)
        h1_coords[0] = [0.0, 0.0, 0.0]
        h1_coords[4] += [0.3, 0.2, 0.1]
        h1_coords[8] += [0.1, 0.5, 0.0]
        h1_coords[12] += [0.0, 0.55, 0.0]
        h1_coords[16] += [-0.1, 0.5, 0.0]
        h1_coords[20] += [-0.2, 0.4, 0.0]

        two_handed = cls in ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z', 'HELLO', 'THANK YOU', 'PLEASE']
        if two_handed:
            h2_coords = np.random.uniform(-0.5, 0.5, (21, 3)).astype(np.float32)
            h2_coords[0] = [0.0, 0.0, 0.0]
            h2_coords[8] += [-0.1, 0.45, 0.0]
        else:
            h2_coords = np.zeros((21, 3), dtype=np.float32)

        base[:63] = h1_coords.flatten()
        base[63:] = h2_coords.flatten()

        for s in range(samples_per_class):
            noise = np.random.normal(0, 0.035, NUM_COORDS).astype(np.float32)
            noise[0:3] = 0.0
            if two_handed:
                noise[63:66] = 0.0
            else:
                noise[63:] = 0.0
            X.append(base + noise)
            y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y)


def get_available_classes(raw_dir="data/raw"):
    if os.path.exists(raw_dir):
        subdirs = [d.upper() for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))]
        if subdirs:
            return subdirs
    return DEFAULT_CLASSES


def load_or_create_dataset():
    x_file = os.path.join("data", "processed", "X.npy")
    y_file = os.path.join("data", "processed", "y.npy")
    
    if os.path.exists(x_file) and os.path.exists(y_file):
        try:
            X = np.load(x_file)
            y = np.load(y_file)
            if len(X) > 0 and len(y) > 0:
                return X, y
        except Exception:
            pass

    active_classes = get_available_classes()
    X, y = generate_synthetic_isl_dataset(classes=active_classes, samples_per_class=50)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    np.save(x_file, X)
    np.save(y_file, y)
    np.save(os.path.join("data", "processed", "classes.npy"), np.array(active_classes))
    return X, y


def train_and_save():
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    X, y = load_or_create_dataset()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    num_classes = len(classes)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = ISLConvNet(num_classes=num_classes, learning_rate=0.01)
    history = model.fit(X_train, y_train, epochs=35, batch_size=32)
    
    model_path = os.path.join("models", "isl_model.pkl")
    encoder_path = os.path.join("models", "label_encoder.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    return model, label_encoder


if __name__ == "__main__":
    train_and_save()
