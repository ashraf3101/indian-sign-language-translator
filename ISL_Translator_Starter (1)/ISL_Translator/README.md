# Indian Sign Language (ISL) Alphabet Translator

A Streamlit starter project for **static ISL fingerspelled alphabet image recognition** and **typed text → alphabet-sign display**.

> Scope: this starter recognizes the alphabet classes represented in the selected dataset. It is not a complete ISL sentence translator. Some signs may be dynamic, and text-to-sign currently spells text letter by letter using sign images you provide.

## Recommended dataset

Use the [RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset). Its repository describes 26 alphabet classes and a `Dataset.zip` organized into training, testing, and validation folders.

1. Download `Dataset.zip` from the repository.
2. Extract it into `data/raw/`.
3. Make sure the extracted structure resembles:

```text
data/raw/
└── Dataset/
    ├── Training (A-Z)/
    │   ├── A/
    │   ├── B/
    │   └── ...
    ├── Testing (A-Z)/
    │   ├── A/
    │   └── ...
    └── Validation (A-Z)/
        ├── A/
        └── ...
```

The script searches for these folder names recursively, so the outer `Dataset/` directory can have a different name. If the dataset uses different folder names, update `find_split_dir()` in `src/train_model.py`.

## 1. Install

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Train locally

Training is done locally, not on Streamlit Cloud:

```bash
python src/train_model.py
```

This saves:
- `models/isl_model.keras`
- `models/labels.json`

The code uses MobileNetV2 transfer learning to keep the model practical for deployment. For best evaluation, use the dataset's separate test split and do not mix images from the same signer across train and test if signer metadata is available.

## 3. Run locally

```bash
streamlit run app.py
```

The app offers:
- **Sign → Text:** upload an image or take a camera snapshot; the model predicts one alphabet class.
- **Text → Sign:** type text; the app spells it letter-by-letter. Add verified ISL alphabet images as `assets/signs/A.png`, `assets/signs/B.png`, etc. Missing sign images are shown as text placeholders.

## 4. Deploy on Streamlit Community Cloud

Push the project to GitHub, including:
- `app.py`
- `src/`
- `models/isl_model.keras`
- `models/labels.json`
- `assets/`
- `requirements.txt`

Do **not** upload the entire raw dataset to GitHub unless you have checked repository/file size limits and the dataset license. Train locally first, then commit the model artifact. If the model file is too large for GitHub, use Git LFS or an approved external model hosting option and modify the loading code accordingly.

On Streamlit Community Cloud, choose `app.py` as the entry point.

## Dataset and responsible use

Check the source repository's current license and attribution requirements before redistributing the dataset or any derived assets. Recognition quality depends on lighting, hand position, camera angle, signer variation, and whether the sign is static. This is a prototype, not a substitute for a qualified interpreter.
