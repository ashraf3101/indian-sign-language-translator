"""
app.py
Main Streamlit application for Indian Sign Language (ISL) Recognition & Translation
Powered by a 1D Convolutional Neural Network (CNN) and MediaPipe Hands.
"""

import os
import io
import time
import pickle
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="ISL Recognition & Translation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .header-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 4px; }
    .header-sub { font-size: 1rem; opacity: 0.9; }
    .pred-box {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 12px 0;
    }
    .pred-label { font-size: 2.8rem; font-weight: 800; color: #1a73e8; }
    .sentence-container {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 1.3rem;
        font-family: monospace;
        min-height: 55px;
        display: flex;
        align-items: center;
    }
    .sign-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Safe imports
try:
    from src.preprocess import extract_landmarks, TOTAL_LANDMARK_FEATURES, MP_AVAILABLE
except Exception:
    MP_AVAILABLE = False
    extract_landmarks = None
    TOTAL_LANDMARK_FEATURES = 126

try:
    from src.train_model import ISLConvNet, train_and_save, DEFAULT_CLASSES
except Exception:
    ISLConvNet = None
    train_and_save = None
    DEFAULT_CLASSES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'HELLO', 'THANK YOU', 'YES', 'NO', 'PLEASE'
    ]

try:
    from src.text_to_sign import text_to_sign_sequence, create_sign_collage
except Exception:
    text_to_sign_sequence = None
    create_sign_collage = None


@st.cache_resource
def load_or_init_model():
    model_path = os.path.join("models", "isl_model.pkl")
    encoder_path = os.path.join("models", "label_encoder.pkl")
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            if hasattr(model, "predict_proba") and hasattr(label_encoder, "classes_"):
                return model, label_encoder
        except Exception:
            pass

    if train_and_save:
        with st.spinner("Initializing ISL CNN model..."):
            model, label_encoder = train_and_save()
            return model, label_encoder
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(DEFAULT_CLASSES)
        m = ISLConvNet(num_classes=len(DEFAULT_CLASSES)) if ISLConvNet else None
        return m, le


if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_predicted" not in st.session_state:
    st.session_state.last_predicted = ""


def speak_via_browser(text):
    if not text:
        return
    clean = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    speech_js = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance('{clean}');
            utterance.rate = 0.95;
            utterance.lang = 'en-IN';
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(speech_js, height=0)


model, label_encoder = load_or_init_model()
classes = list(label_encoder.classes_) if label_encoder and hasattr(label_encoder, 'classes_') else DEFAULT_CLASSES


# --- SIDEBAR ---
with st.sidebar:
    st.title("🤟 ISL Classifier")
    st.markdown("Convolutional Neural Network for Indian Sign Language.")
    st.divider()
    st.markdown(f"**Vocab Size**: `{len(classes)}` signs")
    st.markdown(f"**MediaPipe**: `{'Active' if MP_AVAILABLE else 'Headless'}`")
    st.markdown("**Dual Hands**: Supported (up to 2)")
    conf_threshold = st.slider("Confidence Threshold", 0.30, 0.95, 0.60, 0.05)


# --- HEADER ---
st.markdown("""
<div class="header-card">
    <div class="header-title">🇮🇳 Indian Sign Language Recognition & Translation</div>
    <div class="header-sub">
        Real-time two-handed gesture recognition with Deep CNN & MediaPipe
    </div>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Real-Time Recognition",
    "✍️ Text to Sign",
    "📖 Sign Dictionary",
    "🧠 CNN Architecture"
])

# TAB 1: RECOGNITION
with tab1:
    col_in, col_out = st.columns([1.1, 0.9])
    with col_in:
        mode = st.radio("Input Source:", ["📸 Camera Snapshot", "📁 Upload Image", "🧪 Test Sample Signs"], horizontal=True)
        img_to_process = None

        if mode == "📸 Camera Snapshot":
            cam = st.camera_input("Take a snapshot of your hand gesture:")
            if cam:
                img_to_process = Image.open(cam)
        elif mode == "📁 Upload Image":
            up = st.file_uploader("Upload hand gesture image", type=["jpg", "png", "jpeg"])
            if up:
                img_to_process = Image.open(up)
        else:
            sample_class = st.selectbox("Select sample ISL sign:", classes)
            real_sample = os.path.join("assets", "signs", "samples", f"{sample_class}.jpg")
            card_sample = os.path.join("assets", "signs", f"{sample_class}.png")
            if os.path.exists(real_sample):
                img_to_process = Image.open(real_sample)
                st.info(f"Loaded real dataset image for sign '{sample_class}'.")
            elif os.path.exists(card_sample):
                img_to_process = Image.open(card_sample)
                st.info(f"Loaded reference card for '{sample_class}'.")

    with col_out:
        st.markdown("#### Recognition Output")
        if img_to_process is not None:
            img_np = np.array(img_to_process.convert("RGB"))
            bgr_img = img_np[:, :, ::-1].copy()

            if extract_landmarks:
                features, annotated_bgr, hands_count = extract_landmarks(bgr_img, draw_landmarks=True)
                display_img = annotated_bgr[:, :, ::-1]
            else:
                features = np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32)
                display_img = img_np
                hands_count = 1

            st.image(display_img, caption=f"Identified Hands: {hands_count}", use_container_width=True)

            if model:
                probs = model.predict_proba(features.reshape(1, -1))[0]
                top_idx = int(np.argmax(probs))
                conf = float(probs[top_idx])
                pred = classes[top_idx]

                if conf >= conf_threshold:
                    st.session_state.last_predicted = pred
                    st.markdown(f"""
                    <div class="pred-box">
                        <div class="pred-label">{pred}</div>
                        <div>Confidence: <b>{conf * 100:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(conf, 1.0))
                else:
                    st.warning(f"Low confidence ({conf*100:.1f}%). Adjust hand posture.")

                st.markdown("**Top Predictions:**")
                for rank, idx in enumerate(np.argsort(probs)[-3:][::-1]):
                    st.write(f"{rank + 1}. **{classes[idx]}**: `{probs[idx]*100:.1f}%`")
        else:
            st.info("Take a photo or choose a sample to detect signs.")

    st.divider()
    st.markdown("### 💬 Sentence Builder")
    col_s1, col_s2 = st.columns([1.5, 1])
    with col_s1:
        st.markdown(f'<div class="sentence-container">{st.session_state.sentence or "..."}</div>', unsafe_allow_html=True)
    with col_s2:
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("➕ Add"):
            if st.session_state.last_predicted:
                p = st.session_state.last_predicted
                st.session_state.sentence += p if len(p) == 1 else f" {p} "
                st.rerun()
        if c2.button("␣ Space"):
            st.session_state.sentence += " "
            st.rerun()
        if c3.button("⌫ Del"):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
        if c4.button("🗑️ Clear"):
            st.session_state.sentence = ""
            st.rerun()

    if st.button("🔊 Speak Sentence", type="primary"):
        speak_via_browser(st.session_state.sentence)


# TAB 2: TEXT TO SIGN
with tab2:
    st.markdown("### ✍️ Translate Text into Sign Language")
    txt = st.text_input("Enter text:", value="HELLO ISL")
    if txt and text_to_sign_sequence:
        seq = text_to_sign_sequence(txt, assets_dir="assets/signs")
        valid = [s for s in seq if s["type"] != "space"]
        cols = st.columns(min(len(valid), 6) or 1)
        for i, item in enumerate(valid[:12]):
            with cols[i % len(cols)]:
                if item["image_path"] and os.path.exists(item["image_path"]):
                    st.image(item["image_path"], caption=item["token"], use_container_width=True)
                else:
                    st.markdown(f'<div class="sign-card"><h4>{item["token"]}</h4></div>', unsafe_allow_html=True)


# TAB 3: DICTIONARY
with tab3:
    st.markdown("### 📖 ISL Sign Dictionary")
    cols_per_row = 6
    for i in range(0, len(classes), cols_per_row):
        batch = classes[i:i + cols_per_row]
        cols = st.columns(len(batch))
        for col, c_name in zip(cols, batch):
            with col:
                card_path = os.path.join("assets", "signs", f"{c_name}.png")
                if os.path.exists(card_path):
                    st.image(card_path, caption=c_name, use_container_width=True)
                else:
                    st.markdown(f'<div class="sign-card"><h3>{c_name}</h3></div>', unsafe_allow_html=True)


# TAB 4: CNN ARCHITECTURE
with tab4:
    st.markdown("### 🧠 1D Convolutional Neural Network (CNN)")
    st.code("""
Input (Batch, 42 joints, 3 coords) -> 126 Features
Conv1D (32 filters, k=3) + ReLU + MaxPool(2)
Conv1D (64 filters, k=3) + ReLU + MaxPool(2)
Flatten (576 Features)
Dense (128 units, ReLU) + Dropout(0.3)
Dense Output (Softmax) -> Sign Probabilities
    """, language="text")
    metrics_path = os.path.join("assets", "training_metrics.png")
    if os.path.exists(metrics_path):
        st.image(metrics_path, caption="CNN Convergence Curves", use_container_width=True)
