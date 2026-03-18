"""
train_model.py
==============
Day 2 — CNN Training Script
Driver Drowsiness Detection System

Model   : MobileNetV2 (transfer learning + fine-tuning)
Classes : closed_eye | open_eye | yawn
Dataset : 216,000 train / 27,000 val  (80/10/10 split)
Target  : >92% train acc, >90% val acc, >85% yawn class acc

VRAM optimized for 4 GB (RTX 2050) — batch=16, img=96x96, mixed precision

Author  : Austin Trinidad
"""

import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed during training
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# 0. GPU SETUP  (critical for 4 GB VRAM)
# ─────────────────────────────────────────────

def setup_gpu():
    """Enable GPU memory growth to avoid OOM on 4 GB VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Found {len(gpus)} GPU(s). Memory growth enabled.")
            print(f"[GPU] {gpus[0].name}")
        except RuntimeError as e:
            print(f"[GPU] Memory growth error: {e}")
    else:
        print("[GPU] No GPU found -- training on CPU. This will be very slow.")
        print("[GPU] Check your CUDA / cuDNN installation.")

    # Mixed precision: float16 on GPU, float32 for stability
    policy = keras.mixed_precision.Policy("mixed_float16")
    keras.mixed_precision.set_global_policy(policy)
    print(f"[GPU] Mixed precision policy: {policy.name}")


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "validation")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs", "training")
PLOTS_DIR   = os.path.join(BASE_DIR, "logs", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE     = 96
BATCH_SIZE   = 16
NUM_CLASSES  = 3
CLASS_NAMES  = ["closed_eye", "open_eye", "yawn"]

PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3

# Phase 2 resume — 15 more epochs from the epoch 4 checkpoint (93.05%)
PHASE2_EPOCHS       = 15
PHASE2_LR           = 1e-5
UNFREEZE_FROM_LAYER = 100

MODEL_SAVE_PATH  = os.path.join(MODEL_DIR, "drowsiness_model.keras")
BEST_PHASE1_PATH = os.path.join(MODEL_DIR, "phase1_best.keras")
CSV_LOG_PATH     = os.path.join(LOG_DIR, "training_log.csv")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 2. DATA GENERATORS
# ─────────────────────────────────────────────

def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("\n[DATA] Loading training set...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
    )

    print("[DATA] Loading validation set...")
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print("[DATA] Loading test set...")
    test_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"\n[DATA] Class indices: {train_gen.class_indices}")
    print(f"[DATA] Train samples : {train_gen.samples:,}")
    print(f"[DATA] Val samples   : {val_gen.samples:,}")
    print(f"[DATA] Test samples  : {test_gen.samples:,}")

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"\n[DATA] Class weights: {class_weight_dict}")
    return class_weight_dict


# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    print(f"\n[MODEL] MobileNetV2 total layers: {len(base_model.layers)}")

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)

    x       = layers.Dense(256)(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation("relu")(x)
    x       = layers.Dropout(0.4)(x)

    x       = layers.Dense(128)(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation("relu")(x)
    x       = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, dtype="float32")(x)
    outputs = layers.Activation("softmax", dtype="float32")(outputs)

    model = Model(inputs, outputs, name="drowsiness_detector")
    return model, base_model


# ─────────────────────────────────────────────
# 4. CALLBACKS
# ─────────────────────────────────────────────

def get_phase1_callbacks():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return [
        ModelCheckpoint(
            filepath=BEST_PHASE1_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"phase1_{timestamp}")),
        CSVLogger(CSV_LOG_PATH, append=False),
    ]


def get_phase2_callbacks():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return [
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"phase2_{timestamp}")),
        CSVLogger(CSV_LOG_PATH, append=True),
    ]


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────

def train_phase1(model, train_gen, val_gen, class_weights):
    """Phase 1: freeze MobileNetV2 base, train top layers only."""
    print("\n" + "="*60)
    print("PHASE 1 -- Training classification head (base frozen)")
    print("="*60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        train_gen,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_phase1_callbacks(),
        verbose=1,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\n[PHASE 1] Best val accuracy: {val_acc:.4f}")
    return history


def unfreeze_for_phase2(model, base_model):
    """Unfreeze top layers of MobileNetV2 for fine-tuning."""
    base_model.trainable = True

    for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n[PHASE 2] Unfroze {trainable_count} MobileNetV2 layers (from index {UNFREEZE_FROM_LAYER})")
    print(f"[PHASE 2] All BatchNorm layers remain frozen")


def train_phase2(model, train_gen, val_gen, class_weights):
    """Phase 2: fine-tune top MobileNetV2 layers with very low LR."""
    print("\n" + "="*60)
    print("PHASE 2 -- Fine-tuning (top MobileNetV2 layers unfrozen)")
    print("="*60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        epochs=PHASE2_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_phase2_callbacks(),
        verbose=1,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\n[PHASE 2] Best val accuracy: {val_acc:.4f}")
    return history


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def save_training_plots(history):
    """Save accuracy and loss curves for Phase 2 resume."""

    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc,     "b-", label="Train accuracy")
    ax1.plot(epochs, val_acc, "r-", label="Val accuracy")
    ax1.set_title("Accuracy (Phase 2 resume — starting from 93.05%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss,     "b-", label="Train loss")
    ax2.plot(epochs, val_loss, "r-", label="Val loss")
    ax2.set_title("Loss (Phase 2 resume)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Drowsiness Model -- Phase 2 Resume Curves", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOTS] Training curves saved -> {plot_path}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Driver Drowsiness Detection -- Day 2: Phase 2 Resume")
    print("Resuming from epoch 4 checkpoint — val accuracy: 93.05%")
    print("=" * 60)

    setup_gpu()

    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "validation"), (TEST_DIR, "test")]:
        if not os.path.exists(d):
            print(f"\n[ERROR] {name} directory not found: {d}")
            print("Make sure your dataset/ folder is in the project root.")
            sys.exit(1)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[ERROR] Phase 2 checkpoint not found: {MODEL_SAVE_PATH}")
        print("Expected drowsiness_model.keras from previous Phase 2 run.")
        sys.exit(1)

    train_gen, val_gen, test_gen = build_generators()
    class_weights = get_class_weights(train_gen)

    # Load the best Phase 2 checkpoint (epoch 4, val_acc 93.05%)
    print(f"\n[RESUME] Loading Phase 2 checkpoint...")
    model = keras.models.load_model(MODEL_SAVE_PATH)
    print(f"[RESUME] Loaded: {MODEL_SAVE_PATH}")
    print(f"[RESUME] Resuming fine-tuning for {PHASE2_EPOCHS} more epochs...")

    # Re-get base_model reference for unfreezing
    base_model = model.layers[1]

    # Unfreeze same layers as before and continue fine-tuning
    unfreeze_for_phase2(model, base_model)
    history2 = train_phase2(model, train_gen, val_gen, class_weights)

    # Save final model (ModelCheckpoint already saves best, this saves the last)
    model.save(MODEL_SAVE_PATH)
    print(f"\n[MODEL] Final model saved -> {MODEL_SAVE_PATH}")

    save_training_plots(history2)

    print("\n" + "="*60)
    print("Training complete! Run evaluate_model.py next.")
    print("="*60)


if __name__ == "__main__":
    main()