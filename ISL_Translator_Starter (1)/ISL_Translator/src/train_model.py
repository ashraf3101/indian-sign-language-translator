import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "isl_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 3
SEED = 42

def find_split_dir(split_names):
    """Search recursively for a directory whose name matches one of split_names."""
    wanted = {name.casefold() for name in split_names}
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {RAW_DIR}. Extract the dataset into data/raw/."
        )
    for path in RAW_DIR.rglob("*"):
        if path.is_dir() and path.name.casefold() in wanted:
            return path
    raise FileNotFoundError(
        f"Could not find any of these directories under {RAW_DIR}: {split_names}. "
        "Check the extracted dataset folder names."
    )

def load_split(path, shuffle):
    return keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

def main():
    tf.keras.utils.set_random_seed(SEED)
    train_dir = find_split_dir(["Training (A-Z)", "Training", "train", "Train"])
    val_dir = find_split_dir(["Validation (A-Z)", "Validation", "valid", "val"])
    test_dir = find_split_dir(["Testing (A-Z)", "Testing", "test", "Test"])

    train_ds = load_split(train_dir, shuffle=True)
    val_ds = load_split(val_dir, shuffle=False)
    test_ds = load_split(test_dir, shuffle=False)

    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError(
            "Train/validation/test class folders differ. Ensure all splits contain the same "
            "alphabet class folders with identical names."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    augmentation = keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    base = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    # Match src/predict.py: pixel values in [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True
        ),
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Classes:", class_names)
    print("Training transfer-learning classifier...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    # Optional short fine-tuning of the last layers for adaptation to the dataset.
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks)

    # Reload best checkpoint and evaluate once on the held-out test split.
    best_model = keras.models.load_model(MODEL_PATH)
    loss, accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved labels: {LABELS_PATH}")
    print("Next: run `streamlit run app.py`")

if __name__ == "__main__":
    main()
