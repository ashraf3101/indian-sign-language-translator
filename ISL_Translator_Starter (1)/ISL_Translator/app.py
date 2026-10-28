import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_artifacts, predict_image
from src.text_to_sign import render_spelling

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
SIGNS_DIR = ROOT / "assets" / "signs"

st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

st.title("🤟 Indian Sign Language Alphabet Translator")
st.caption("Prototype: static alphabet recognition and letter-by-letter text spelling.")

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    return load_artifacts(MODEL_PATH, LABELS_PATH)

model, labels = get_model()

with st.sidebar:
    st.header("About")
    st.write(
        "This prototype recognizes alphabet classes from a trained image model. "
        "Text-to-sign spells text using alphabet sign images you provide."
    )
    st.warning(
        "It does not yet translate full sentences into grammatical Indian Sign Language. "
        "Some signs may require motion and cannot be reliably identified from one image."
    )

tab_sign, tab_text = st.tabs(["📷 Sign → Text", "⌨️ Text → Sign"])

with tab_sign:
    st.subheader("Recognize one alphabet sign")
    if model is None:
        st.error(
            "Model files are missing. Train the model locally with "
            "`python src/train_model.py`, then place `isl_model.keras` and `labels.json` "
            "inside the `models/` folder before launching or deploying the app."
        )
    else:
        source = st.radio("Input method", ["Upload an image", "Take a camera snapshot"], horizontal=True)
        uploaded = None
        if source == "Upload an image":
            uploaded = st.file_uploader("Upload a hand-sign image", type=["png", "jpg", "jpeg"])
        else:
            uploaded = st.camera_input("Take a clear picture of the sign")

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Input image", width=360)
                with st.spinner("Recognizing sign..."):
                    label, confidence, top_predictions = predict_image(image, model, labels)
                st.success(f"Predicted sign: **{label}**")
                st.metric("Model confidence", f"{confidence:.1%}")
                with st.expander("Top predictions"):
                    for name, score in top_predictions:
                        st.write(f"{name}: {score:.1%}")
                        st.progress(max(0.0, min(1.0, float(score))))
                st.info(
                    "Check the prediction before using it. Confidence is a model score, "
                    "not a guarantee of correctness."
                )
            except Exception as exc:
                st.error(f"Could not process this image: {exc}")

with tab_text:
    st.subheader("Spell text using ISL alphabet signs")
    text = st.text_input("Enter text", placeholder="For example: HELLO")
    if st.button("Show sign sequence", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            render_spelling(text, SIGNS_DIR)
