"""
evaluate_model.py
=================
Day 2 — Model Evaluation Script
Driver Drowsiness Detection System

Loads the trained model and evaluates on the 27,000 test images.
Outputs:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Misclassification analysis

Run AFTER train_model.py has completed.

Author: Austin Trinidad
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)

# ─────────────────────────────────────────────
# CONFIG — must match train_model.py
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR    = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "drowsiness_model.keras")
PLOTS_DIR   = os.path.join(BASE_DIR, "logs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE    = 96
BATCH_SIZE  = 16
CLASS_NAMES = ["closed_eye", "open_eye", "yawn"]


# ─────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] GPU enabled: {gpus[0].name}")
    else:
        print("[GPU] No GPU — running on CPU")


# ─────────────────────────────────────────────
# LOAD MODEL + DATA
# ─────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("Run train_model.py first.")
        raise FileNotFoundError(MODEL_PATH)

    print(f"[MODEL] Loading from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("[MODEL] Loaded successfully.")
    return model


def load_test_generator():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,  # MUST be False for label alignment
    )
    print(f"[DATA] Test samples: {test_gen.samples:,}")
    print(f"[DATA] Class indices: {test_gen.class_indices}")
    return test_gen


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def run_evaluation(model, test_gen):
    """Get predictions and true labels for all test images."""
    print("\n[EVAL] Running predictions on test set...")
    print("[EVAL] This may take a few minutes...")

    # Predict in batches
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    return y_true, y_pred, y_pred_probs


def print_metrics(y_true, y_pred):
    """Print overall accuracy and per-class classification report."""
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "="*60)
    print(f"OVERALL TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("="*60)

    # Check against Day 2 targets
    targets = {"overall": 0.90, "yawn_recall": 0.85}
    status = "PASS" if acc >= targets["overall"] else "BELOW TARGET"
    print(f"Target: >90% val accuracy -> {status}")

    print("\nPER-CLASS REPORT:")
    print("-"*60)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    # Highlight yawn class specifically (hardest class)
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None)
    yawn_idx = CLASS_NAMES.index("yawn")
    yawn_recall = recalls[yawn_idx]
    yawn_status = "PASS" if yawn_recall >= 0.85 else "BELOW TARGET (needs improvement)"
    print(f"Yawn recall: {yawn_recall:.4f} ({yawn_recall*100:.2f}%) -> {yawn_status}")

    return acc


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred):
    """Save a clean, annotated confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalised (row = recall per class)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title("Confusion Matrix (normalised — row = recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("Drowsiness Model -- Test Set Confusion Matrix", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOTS] Confusion matrix saved -> {save_path}")

    # Print raw matrix to console too
    print("\nRaw confusion matrix:")
    header = f"{'':15}" + "".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i]:15}" + "".join(f"{v:>12}" for v in row)
        print(row_str)


# ─────────────────────────────────────────────
# CONFIDENCE ANALYSIS
# ─────────────────────────────────────────────

def confidence_analysis(y_true, y_pred, y_pred_probs):
    """Show average confidence per class and flag low-confidence predictions."""
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)

    for i, cls in enumerate(CLASS_NAMES):
        mask = (y_true == i)
        if mask.sum() == 0:
            continue
        probs = y_pred_probs[mask, i]
        print(f"{cls:>12}: mean confidence = {probs.mean():.4f}  "
              f"min = {probs.min():.4f}  max = {probs.max():.4f}")

    # Low-confidence predictions (model is uncertain)
    max_probs = y_pred_probs.max(axis=1)
    low_conf = (max_probs < 0.70).sum()
    pct = low_conf / len(y_true) * 100
    print(f"\nLow-confidence predictions (<70%): {low_conf:,} / {len(y_true):,} ({pct:.1f}%)")
    print("(These will trigger 'uncertain' handling in real-time detection)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Driver Drowsiness Detection -- Day 2: Evaluation")
    print("=" * 60)

    setup_gpu()

    model    = load_model()
    test_gen = load_test_generator()

    y_true, y_pred, y_pred_probs = run_evaluation(model, test_gen)

    acc = print_metrics(y_true, y_pred)

    save_confusion_matrix(y_true, y_pred)
    confidence_analysis(y_true, y_pred, y_pred_probs)

    print("\n" + "="*60)
    if acc >= 0.90:
        print(f"Day 2 COMPLETE. Model ready for Day 3 (real-time detection).")
    else:
        print(f"Accuracy {acc:.2%} is below 90% target.")
        print("Consider: more fine-tuning epochs, lower LR, or data audit.")
    print(f"Model: {MODEL_PATH}")
    print(f"Plots: {PLOTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()