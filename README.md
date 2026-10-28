# 🇮🇳 Indian Sign Language (ISL) Recognition System using CNN

A production-ready Deep Learning platform for **Indian Sign Language (ISL)** recognition, translation, and learning. Built with a **Convolutional Neural Network (CNN)** and **MediaPipe Hands**, featuring two-handed gesture support, real-time sentence building, browser-based voice synthesis (Text-to-Speech), and an English-to-ISL visual translator.

Ready for one-click deployment on **Streamlit Community Cloud** or local execution.

---

## 📁 Project Structure

The project strictly follows the required format:

```text
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Python library dependencies
├── packages.txt            # System dependencies for Streamlit Cloud (Linux)
│
├── data/
│   ├── raw/                # Original ISL gesture images (Classes A-Z, 12,600+ samples)
│   └── processed/          # Extracted 126-D hand landmarks and labels (X.npy, y.npy)
│
├── models/
│   ├── isl_model.pkl       # Trained 1D-CNN classifier model
│   └── label_encoder.pkl   # Serialized sign class labels
│
├── src/
│   ├── collect_data.py     # Collect webcam gesture samples
│   ├── preprocess.py       # Extract & normalize dual-hand landmarks
│   ├── train_model.py      # Define and train the CNN classifier
│   ├── predict.py          # Real-time desktop webcam predictor
│   └── text_to_sign.py     # English text to ISL visual sign translation
│
└── assets/
    ├── signs/              # ISL sign card visual assets & reference samples
    ├── videos/             # Sign demonstration videos
    └── training_metrics.png# CNN loss and accuracy convergence plot
```

---

## 🚀 Streamlit Cloud Deployment Guide

Deploy this application live on the web for free using [Streamlit Community Cloud](https://share.streamlit.io):

### Step 1: Push Code to GitHub
1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Indian Sign Language CNN project"
   ```
2. Create a new repository on [GitHub](https://github.com/new).
3. Push your repository:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, set **Branch** to `main`, and **Main file path** to `app.py`.
4. Click **Deploy!**
5. Streamlit Cloud will automatically install dependencies from `requirements.txt` and `packages.txt` and launch your app.

---

## 💻 Local Setup & Execution

### 1. Prerequisites
Ensure Python (version 3.9, 3.10, or 3.11 recommended) is installed on your machine.

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd Indian_sign_language

# Create virtual environment
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ CLI Utilities & Workflow

### 1. Real-Time Desktop Sign Predictor
Predict ISL signs using your local webcam with HUD overlay, temporal smoothing, and audio announcements:
```bash
python src/predict.py
```
- **Hold sign steady** to commit letters/words into a sentence.
- `[Space]` to add space.
- `[d]` or `[Backspace]` to delete last character.
- `[c]` to clear sentence.
- `[s]` to speak sentence aloud.
- `[q]` to quit.

### 2. Collect New Webcam Samples
Capture custom gesture samples to expand the vocabulary:
```bash
python src/collect_data.py --label NAMASTE --samples 100
```

### 3. Preprocess Raw Dataset
Extract 126-D normalized MediaPipe landmarks for Hand 1 and Hand 2:
```bash
python src/preprocess.py
```

### 4. Train the CNN Classifier
Train the 1D Convolutional Neural Network on the extracted features:
```bash
python src/train_model.py
```
Saves the updated `models/isl_model.pkl` and `models/label_encoder.pkl`.

### 5. Text to Indian Sign Language Translator
Translate English text into an ISL sign sequence and save a stitched collage:
```bash
python src/text_to_sign.py --text "HELLO INDIA" --save assets/translation_output.png
```

---

## 🧠 CNN Architecture Details

Indian Sign Language (ISL) frequently uses **two hands simultaneously** (e.g. letters `B`, `D`, `F`, `G`, `H`, `K`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `W`, `X`, `Y`, `Z`).

### Feature Representation:
- **Max Hands**: 2 hands tracked simultaneously by MediaPipe Hands.
- **Landmarks**: 21 3D joint coordinates $(x, y, z)$ per hand = 42 joints $\times$ 3 coordinates = **126 features**.
- **Normalization**:
  - Wrist centered at $(0, 0, 0)$ for translation invariance.
  - Scaled by maximum palm/joint distance for scale and distance invariance.

### Convolutional Neural Network:
- **Input**: `(Batch, 42, 3)` sequence of hand joint coordinates.
- **Conv1D Layer 1**: 32 filters, kernel size 3, ReLU, Batch Normalization, MaxPooling1D (pool=2).
- **Conv1D Layer 2**: 64 filters, kernel size 3, ReLU, Batch Normalization, MaxPooling1D (pool=2).
- **Flatten Layer**: 576-dimensional joint feature representation.
- **Dense Layer**: 128 hidden units, ReLU, Dropout (0.30).
- **Output Layer**: Dense with Softmax over ISL classes.
- **Inference Latency**: Under 5ms on standard CPU.

---

## 📜 Dataset Information

The raw dataset (`data/raw/`) includes 12,600+ high-quality images across 26 ISL alphabet classes (`A` through `Z`).
Each class contains ~480 variations captured across multiple lighting conditions and hand angles.
