"""
Indian Sign Language Recognition & Translation
Streamlit app with safe, lazy model loading for Streamlit Community Cloud.
"""

from pathlib import Path
import json
import pickle
import html
import sys
import numpy as np
import streamlit as st
from PIL import Image

# Resolve paths relative to this app.py, not the process working directory.
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "isl_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

st.set_page_config(
    page_title="ISL Recognition & Translation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 22px; border-radius: 12px; margin-bottom: 20px;
    }
    .header-title { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
    .header-sub { font-size: 1rem; opacity: .92; }
    .pred-box {
        background: #f8f9fa; border: 2px solid #e9ecef; border-radius: 12px;
        padding: 18px; text-align: center; margin: 12px 0;
    }
    .pred-label { font-size: 2.5rem; font-weight: 800; color: #1a73e8; }
    .sentence-container {
        background: #fff; border: 2px solid #e0e0e0; border-radius: 8px;
        padding: 14px 18px; font-size: 1.2rem; font-family: monospace;
        min-height: 55px; overflow-wrap: anywhere;
    }
    .sign-card {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;
        text-align: center; background: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep imports separate so a missing optional dependency does not prevent
# the rest of the interface from rendering.
MP_AVAILABLE = False
extract_landmarks = None
TOTAL_LANDMARK_FEATURES = 126
preprocess_error = None

try:
    # Support both layouts: src/preprocess.py or preprocess.py beside app.py.
    try:
        from src.preprocess import (
            extract_landmarks,
            TOTAL_LANDMARK_FEATURES,
            MP_AVAILABLE,
        )
    except ModuleNotFoundError as import_exc:
        if import_exc.name != "src":
            raise
        from preprocess import (
            extract_landmarks,
            TOTAL_LANDMARK_FEATURES,
            MP_AVAILABLE,
        )
except Exception as exc:
    preprocess_error = f"{type(exc).__name__}: {exc}"

text_to_sign_sequence = None
text_to_sign_error = None
try:
    from src.text_to_sign import text_to_sign_sequence
except Exception as exc:
    text_to_sign_error = f"{type(exc).__name__}: {exc}"

ISLConvNet = None
train_and_save = None
DEFAULT_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "HELLO", "THANK YOU", "YES", "NO", "PLEASE",
]
train_model_error = None
try:
    from src.train_model import ISLConvNet, train_and_save, DEFAULT_CLASSES
except Exception as exc:
    train_model_error = f"{type(exc).__name__}: {exc}"

if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_predicted" not in st.session_state:
    st.session_state.last_predicted = ""
if "isl_model" not in st.session_state:
    st.session_state.isl_model = None
if "isl_encoder" not in st.session_state:
    st.session_state.isl_encoder = None
if "model_error" not in st.session_state:
    st.session_state.model_error = None


@st.cache_resource(show_spinner=False)
def load_saved_model(model_file: str, encoder_file: str):
    """Load the committed model and encoder, with compatibility for old pickles."""
    # Some versions of the training script may have pickled the custom class as
    # __main__.ISLConvNet. Make that class available before unpickling.
    if ISLConvNet is not None:
        setattr(sys.modules["__main__"], "ISLConvNet", ISLConvNet)

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(encoder_file, "rb") as f:
        encoder = pickle.load(f)

    # Restore methods if this is an older instance of our custom network whose
    # class was pickled from a different module path. Only do this when the
    # expected network weights are present; never rebind an arbitrary object.
    required_weights = (
        "W_conv1", "b_conv1", "W_conv2", "b_conv2",
        "W_dense1", "b_dense1", "W_out", "b_out",
    )
    if not callable(getattr(model, "predict", None)) and ISLConvNet is not None:
        if all(hasattr(model, attr) for attr in required_weights):
            try:
                model.__class__ = ISLConvNet
            except (TypeError, AttributeError):
                pass

    if not callable(getattr(model, "predict", None)):
        model_type = f"{type(model).__module__}.{type(model).__name__}"
        available = [
            name for name in ("predict", "predict_proba", "fit", "forward")
            if callable(getattr(model, name, None))
        ]
        raise TypeError(
            f"Saved model type is {model_type} and has no usable predict() method. "
            f"Detected methods: {available or 'none'}. The model file may be from "
            "a different/older training version; regenerate both model files from "
            "the current src/train_model.py and commit them together."
        )

    if not hasattr(encoder, "classes_"):
        raise TypeError("Saved label encoder does not provide classes_.")

    # Custom ISLConvNet predicts integer class indices and has no classes_ field.
    # Ensure the encoder and model output sizes are compatible where detectable.
    model_class_count = getattr(model, "num_classes", None)
    if model_class_count is not None and int(model_class_count) != len(encoder.classes_):
        raise ValueError(
            f"Model has {model_class_count} output classes, but label encoder has "
            f"{len(encoder.classes_)} classes. Replace both saved files from the same training run."
        )
    return model, encoder


def load_model_for_session():
    """Load saved model on demand, with a clear message if artifacts are absent."""
    if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
        missing = []
        if not MODEL_PATH.is_file():
            missing.append(str(MODEL_PATH.relative_to(BASE_DIR)))
        if not ENCODER_PATH.is_file():
            missing.append(str(ENCODER_PATH.relative_to(BASE_DIR)))
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Add the trained model files to the repository, or train a model "
              "locally and commit the generated files."
        )
    return load_saved_model(str(MODEL_PATH), str(ENCODER_PATH))


def speak_via_browser(text):
    if not text.strip():
        st.info("Add a sign to the sentence first.")
        return
    # json.dumps safely escapes quotes and newlines in the text.
    text_json = json.dumps(text)
    speech_js = f"""
    const phrase = {text_json};
    if ("speechSynthesis" in window) {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(phrase);
        utterance.rate = 0.95;
        utterance.lang = "en-IN";
        window.speechSynthesis.speak(utterance);
    }}
    """
    st.components.v1.html(speech_js, height=0)


# Header is rendered before model loading so the app remains responsive.
st.markdown(
    """
    <div class="header-card">
      <div class="header-title">🇮🇳 Indian Sign Language Recognition &amp; Translation</div>
      <div class="header-sub">Hand-gesture recognition, text-to-sign references, and sign dictionary</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🤟 ISL Classifier")
    st.write("Indian Sign Language recognition and translation.")
    st.divider()
    st.caption(f"Python files directory: `{BASE_DIR.name}`")
    st.caption(f"MediaPipe import: {'Available' if MP_AVAILABLE else 'Unavailable'}")
    if preprocess_error:
        with st.expander("Preprocessing import details"):
            st.code(preprocess_error)
    if train_model_error:
        with st.expander("Model module import details"):
            st.code(train_model_error)
    if st.button("Load trained model", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading trained model files..."):
                st.session_state.isl_model, st.session_state.isl_encoder = load_model_for_session()
                st.session_state.model_error = None
            st.success("Model loaded.")
        except Exception as exc:
            st.session_state.isl_model = None
            st.session_state.isl_encoder = None
            st.session_state.model_error = f"{type(exc).__name__}: {exc}"

model = st.session_state.isl_model
label_encoder = st.session_state.isl_encoder
classes = (
    list(label_encoder.classes_)
    if label_encoder is not None and hasattr(label_encoder, "classes_")
    else list(DEFAULT_CLASSES)
)

if st.session_state.model_error:
    st.error("The trained model could not be loaded.")
    st.code(st.session_state.model_error)
elif model is None:
    st.info(
        "The interface is ready. Click **Load trained model** in the sidebar "
        "to enable gesture predictions. Model loading is intentionally delayed "
        "so it cannot block the initial page render."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["📷 Recognition", "✍️ Text to Sign", "📖 Sign Dictionary", "🧠 CNN Architecture"]
)

with tab1:
    col_in, col_out = st.columns([1.1, 0.9])
    with col_in:
        mode = st.radio(
            "Input source:",
            ["📸 Camera Snapshot", "📁 Upload Image", "🧪 Test Sample Signs"],
            horizontal=True,
        )
        img_to_process = None

        if mode == "📸 Camera Snapshot":
            cam = st.camera_input("Take a snapshot of your hand gesture")
            if cam is not None:
                try:
                    img_to_process = Image.open(cam).convert("RGB")
                except Exception as exc:
                    st.error(f"Could not read the camera image: {exc}")
        elif mode == "📁 Upload Image":
            up = st.file_uploader("Upload a hand gesture image", type=["jpg", "jpeg", "png"])
            if up is not None:
                try:
                    img_to_process = Image.open(up).convert("RGB")
                except Exception as exc:
                    st.error(f"Could not read the uploaded image: {exc}")
        else:
            sample_class = st.selectbox("Select sample ISL sign:", classes)
            sample_paths = [
                ASSETS_DIR / "signs" / "samples" / f"{sample_class}.jpg",
                ASSETS_DIR / "signs" / f"{sample_class}.png",
            ]
            sample_path = next((p for p in sample_paths if p.is_file()), None)
            if sample_path:
                try:
                    img_to_process = Image.open(sample_path).convert("RGB")
                    st.info(f"Loaded reference image for '{sample_class}'.")
                except Exception as exc:
                    st.error(f"Could not read sample image: {exc}")
            else:
                st.info(f"No sample image found for '{sample_class}'. You can upload one instead.")

    with col_out:
        st.markdown("#### Recognition output")
        if img_to_process is None:
            st.info("Take a snapshot, upload an image, or choose a sample sign.")
        else:
            st.image(img_to_process, caption="Input image", use_container_width=True)
            if model is None:
                st.warning("Load a trained model from the sidebar to predict this sign.")
            elif extract_landmarks is None or not MP_AVAILABLE:
                st.error(
                    "Landmark extraction is unavailable. Check the MediaPipe installation "
                    "and the import details in the sidebar."
                )
            else:
                try:
                    rgb = np.asarray(img_to_process.convert("RGB"))
                    bgr = rgb[:, :, ::-1].copy()
                    features, annotated_bgr, hands_count = extract_landmarks(
                        bgr, draw_landmarks=True
                    )
                    if features is None or hands_count == 0:
                        st.warning("No hand landmarks were detected. Try a clearer image with your hand fully visible.")
                    else:
                        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
                        expected = getattr(model, "n_features_in_", features.shape[1])
                        if features.shape[1] != expected:
                            st.error(
                                f"Model expects {expected} features, but preprocessing returned "
                                f"{features.shape[1]}. The model and preprocessing code must match."
                            )
                        else:
                            if annotated_bgr is not None:
                                st.image(
                                    np.asarray(annotated_bgr)[:, :, ::-1],
                                    caption=f"Detected hands: {hands_count}",
                                    use_container_width=True,
                                )

                            # Our custom ISLConvNet returns encoded integer class IDs.
                            raw_prediction = np.asarray(model.predict(features)).reshape(-1)[0]
                            model_classes = getattr(model, "classes_", None)

                            if isinstance(raw_prediction, (int, np.integer)):
                                raw_index = int(raw_prediction)
                                # For this project's custom network, predictions are
                                # indices into LabelEncoder.classes_.
                                if 0 <= raw_index < len(label_encoder.classes_):
                                    prediction = label_encoder.inverse_transform([raw_index])[0]
                                else:
                                    prediction = str(raw_index)
                            else:
                                prediction = str(raw_prediction)

                            st.markdown(
                                f'<div class="pred-box"><div class="pred-label">'
                                f'{html.escape(str(prediction))}</div></div>',
                                unsafe_allow_html=True,
                            )
                            st.session_state.last_predicted = str(prediction)

                            if callable(getattr(model, "predict_proba", None)):
                                probs = np.asarray(model.predict_proba(features)[0], dtype=float)
                                if probs.size:
                                    confidence_idx = int(np.argmax(probs))
                                    confidence = float(probs[confidence_idx])
                                    st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
                                    st.progress(float(np.clip(confidence, 0, 1)))
                                    st.markdown("**Top predictions**")
                                    top_indices = np.argsort(probs)[-3:][::-1]
                                    for rank, idx in enumerate(top_indices, start=1):
                                        if model_classes is not None and int(idx) < len(model_classes):
                                            class_value = model_classes[int(idx)]
                                            if isinstance(class_value, (int, np.integer)):
                                                try:
                                                    name = label_encoder.inverse_transform([int(class_value)])[0]
                                                except Exception:
                                                    name = class_value
                                            else:
                                                name = class_value
                                        else:
                                            name = label_encoder.classes_[int(idx)] if int(idx) < len(label_encoder.classes_) else str(idx)
                                        st.write(f"{rank}. **{name}** — {probs[int(idx)] * 100:.1f}%")
                            else:
                                st.info("Prediction completed. This model does not provide probability-based confidence scores.")
                except Exception as exc:
                    st.error("Prediction failed. Check the preprocessing/model compatibility.")
                    st.exception(exc)

    st.divider()
    st.markdown("### 💬 Sentence builder")
    st.markdown(
        f'<div class="sentence-container">{html.escape(st.session_state.sentence or "...")}</div>',
        unsafe_allow_html=True,
    )
    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("➕ Add sign", use_container_width=True):
        if st.session_state.last_predicted:
            token = st.session_state.last_predicted
            if len(token) == 1:
                st.session_state.sentence += token
            else:
                if st.session_state.sentence and not st.session_state.sentence.endswith(" "):
                    st.session_state.sentence += " "
                st.session_state.sentence += token
            st.rerun()
        else:
            st.warning("Make a prediction first.")
    if b2.button("␣ Space", use_container_width=True):
        st.session_state.sentence += " "
        st.rerun()
    if b3.button("⌫ Delete", use_container_width=True):
        st.session_state.sentence = st.session_state.sentence[:-1]
        st.rerun()
    if b4.button("🗑️ Clear", use_container_width=True):
        st.session_state.sentence = ""
        st.rerun()
    if b5.button("🔊 Speak", use_container_width=True):
        speak_via_browser(st.session_state.sentence)

with tab2:
    st.markdown("### ✍️ Translate text into sign references")
    text_input = st.text_input("Enter text:", value="HELLO ISL")
    if text_to_sign_sequence is None:
        st.warning("Text-to-sign module could not be imported.")
        if text_to_sign_error:
            st.code(text_to_sign_error)
    elif text_input.strip():
        try:
            sequence = text_to_sign_sequence(
                text_input, assets_dir=str(ASSETS_DIR / "signs")
            )
            valid_items = [item for item in sequence if item.get("type") != "space"]
            if not valid_items:
                st.info("No sign references were returned for this text.")
            else:
                cols = st.columns(min(len(valid_items), 6))
                for i, item in enumerate(valid_items[:12]):
                    with cols[i % len(cols)]:
                        image_path = item.get("image_path")
                        token = str(item.get("token", "Unknown"))
                        if image_path:
                            path = Path(image_path)
                            if not path.is_absolute():
                                path = BASE_DIR / path
                            if path.is_file():
                                st.image(str(path), caption=token, use_container_width=True)
                            else:
                                st.markdown(
                                    f'<div class="sign-card"><h4>{html.escape(token)}</h4></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f'<div class="sign-card"><h4>{html.escape(token)}</h4></div>',
                                unsafe_allow_html=True,
                            )
        except Exception as exc:
            st.error("Text-to-sign translation failed.")
            st.exception(exc)

with tab3:
    st.markdown("### 📖 ISL sign dictionary")
    cols_per_row = 6
    for start in range(0, len(classes), cols_per_row):
        batch = classes[start:start + cols_per_row]
        cols = st.columns(len(batch))
        for col, class_name in zip(cols, batch):
            with col:
                card_path = ASSETS_DIR / "signs" / f"{class_name}.png"
                if card_path.is_file():
                    st.image(str(card_path), caption=class_name, use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="sign-card"><h3>{html.escape(str(class_name))}</h3></div>',
                        unsafe_allow_html=True,
                    )

with tab4:
    st.markdown("### 🧠 1D Convolutional Neural Network")
    st.code(
        """
Input: (batch, 126 landmark features)
Conv1D (32 filters, kernel=3) + ReLU + MaxPool
Conv1D (64 filters, kernel=3) + ReLU + MaxPool
Flatten
Dense (128 units, ReLU) + Dropout
Output: class probabilities
        """.strip(),
        language="text",
    )
    metrics_path = ASSETS_DIR / "training_metrics.png"
    if metrics_path.is_file():
        st.image(str(metrics_path), caption="Training metrics", use_container_width=True)

with st.expander("Deployment diagnostics"):
    st.write("App directory:", str(BASE_DIR))
    st.write("Model file exists:", MODEL_PATH.is_file())
    st.write("Label encoder exists:", ENCODER_PATH.is_file())
    st.write("Preprocessing import successful:", extract_landmarks is not None)
    st.write("MediaPipe available:", MP_AVAILABLE)
    st.write("Text-to-sign import successful:", text_to_sign_sequence is not None)
    st.write("Model module import successful:", ISLConvNet is not None)
