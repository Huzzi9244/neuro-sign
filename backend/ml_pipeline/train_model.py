"""
Neuro-Sign | Phase 2: Train LSTM Model
=======================================
Loads collected gesture sequences from data/ and trains an LSTM classifier.

Usage:
    python train_model.py

Output:
    models/gesture_model.keras      ← trained model
    models/gesture_labels.json      ← label → index mapping
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_SCRIPT_DIR, "data")
MODEL_DIR      = os.path.join(_SCRIPT_DIR, "models")
MODEL_PATH     = os.path.join(MODEL_DIR, "gesture_model.keras")
LABELS_PATH    = os.path.join(MODEL_DIR, "gesture_labels.json")

SEQUENCE_LENGTH = 30    # frames per sequence (must match collect_data.py)
NUM_FEATURES    = 126   # 2 hands × 21 landmarks × 3 coords

EPOCHS          = 100
BATCH_SIZE      = 32
TEST_SPLIT      = 0.15
VAL_SPLIT       = 0.15
RANDOM_STATE    = 42


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset():
    """
    Loads all sequences from data/<GestureName>/sequence_*.npy
    Returns:
        X: np.ndarray of shape (num_samples, SEQUENCE_LENGTH, NUM_FEATURES)
        y: np.ndarray of shape (num_samples,) — integer labels
        label_map: dict {gesture_name: int}
    """
    sequences = []
    labels    = []
    label_map = {}

    gesture_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not gesture_folders:
        raise FileNotFoundError(
            f"No gesture folders found in {DATA_DIR}.\n"
            "Run collect_data.py first to record some gestures."
        )

    for idx, gesture in enumerate(gesture_folders):
        label_map[gesture] = idx
        gesture_dir = os.path.join(DATA_DIR, gesture)

        npy_files = [
            f for f in os.listdir(gesture_dir)
            if f.endswith(".npy")
        ]

        print(f"  [{idx}] {gesture:20s}  →  {len(npy_files)} sequences")

        for npy_file in npy_files:
            seq = np.load(os.path.join(gesture_dir, npy_file))
            # Validate shape
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                sequences.append(seq)
                labels.append(idx)
            else:
                print(f"    [WARN] Skipping {npy_file} — unexpected shape {seq.shape}")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)

    return X, y, label_map


# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
def build_lstm_model(num_classes: int) -> keras.Model:
    """
    Simple stacked LSTM → Dense classifier.
    Input:  (batch, 30, 126)
    Output: (batch, num_classes) softmax probabilities
    """
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 56)
    print("  Neuro-Sign  |  LSTM Model Training")
    print("=" * 56 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("[1/4]  Loading dataset...\n")
    X, y, label_map = load_dataset()
    num_classes = len(label_map)

    print(f"\n  Total samples : {len(X)}")
    print(f"  Gesture count : {num_classes}")
    print(f"  Input shape   : {X.shape}\n")

    if len(X) < 10:
        print("[WARN]  Very few samples. Consider recording more sequences.\n")

    # ── Train/val/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"  Train : {len(X_train)}  |  Val : {len(X_val)}  |  Test : {len(X_test)}\n")

    # ── Build model ────────────────────────────────────────────────────────
    print("[2/4]  Building LSTM model...\n")
    model = build_lstm_model(num_classes)
    model.summary()

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/4]  Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4]  Evaluating on test set...\n")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%\n")

    # ── Save model + labels ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[SAVED]  Model   →  {MODEL_PATH}")

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[SAVED]  Labels  →  {LABELS_PATH}")

    print("\n" + "=" * 56)
    print("  Training complete!")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
