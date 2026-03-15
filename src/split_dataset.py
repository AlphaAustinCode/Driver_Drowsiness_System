"""
split_dataset.py — Train / Validation / Test Split
Driver Drowsiness Detection System

Splits the balanced 270,000 frame dataset into:
  train/      → 80%  (72,000 per class)
  validation/ → 10%  ( 9,000 per class)
  test/       → 10%  ( 9,000 per class)

Moves files (does NOT copy) to save disk space.

Usage:
    python src/split_dataset.py
"""

import os
import shutil
import random

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_PATH = r"D:\Projects\Driver_Drowsiness_System\dataset"
TRAIN_PATH   = os.path.join(DATASET_PATH, "train")
VAL_PATH     = os.path.join(DATASET_PATH, "validation")
TEST_PATH    = os.path.join(DATASET_PATH, "test")

CLASSES      = ["closed_eye", "open_eye", "yawn"]

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# TEST_RATIO = 0.10  (remainder)

SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def list_images(folder: str) -> list:
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]

def progress_bar(done: int, total: int, width: int = 25) -> str:
    pct  = done / total if total > 0 else 0
    fill = int(pct * width)
    bar  = "█" * fill + "░" * (width - fill)
    return f"[{bar}] {pct*100:5.1f}%  {done:,}/{total:,}"

def move_files(files: list, src_dir: str, dst_dir: str, label: str):
    os.makedirs(dst_dir, exist_ok=True)
    for i, fname in enumerate(files, 1):
        shutil.move(
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, fname),
        )
        if i % 2000 == 0 or i == len(files):
            print(f"      {progress_bar(i, len(files))}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'='*58}")
    print(f"  Dataset Split — 80 / 10 / 10")
    print(f"  Dataset : {DATASET_PATH}")
    print(f"{'='*58}\n")

    summary = {}

    for cls in CLASSES:
        src_dir = os.path.join(TRAIN_PATH, cls)

        if not os.path.exists(src_dir):
            print(f"  ⚠  Skipping {cls} — folder not found: {src_dir}")
            continue

        all_files = list_images(src_dir)
        random.shuffle(all_files)
        total = len(all_files)

        # Calculate split indices
        val_start  = int(total * TRAIN_RATIO)
        test_start = int(total * (TRAIN_RATIO + VAL_RATIO))

        train_files = all_files[:val_start]
        val_files   = all_files[val_start:test_start]
        test_files  = all_files[test_start:]

        print(f"  ── {cls} ({total:,} total) ──────────────────")
        print(f"     train      : {len(train_files):,}  (staying in train/)")
        print(f"     validation : {len(val_files):,}  → moving...")
        move_files(val_files, src_dir, os.path.join(VAL_PATH, cls),  cls)

        print(f"     test       : {len(test_files):,}  → moving...")
        move_files(test_files, src_dir, os.path.join(TEST_PATH, cls), cls)

        summary[cls] = {
            "train" : len(train_files),
            "val"   : len(val_files),
            "test"  : len(test_files),
            "total" : total,
        }
        print()

    # ── FINAL SUMMARY ──────────────────────────
    print(f"\n{'='*58}")
    print(f"  SPLIT COMPLETE")
    print(f"{'='*58}")
    print(f"  {'Class':<14} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'─'*50}")

    t_train = t_val = t_test = t_total = 0
    for cls, s in summary.items():
        print(f"  {cls:<14} {s['train']:>8,} {s['val']:>8,} {s['test']:>8,} {s['total']:>8,}")
        t_train += s['train']
        t_val   += s['val']
        t_test  += s['test']
        t_total += s['total']

    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<14} {t_train:>8,} {t_val:>8,} {t_test:>8,} {t_total:>8,}")

    print(f"\n  Folder structure:")
    print(f"    dataset/train/      → {t_train:,} images")
    print(f"    dataset/validation/ → {t_val:,} images")
    print(f"    dataset/test/       → {t_test:,} images")
    print(f"\n  ✅ Ready for CNN training!\n")


if __name__ == "__main__":
    main()