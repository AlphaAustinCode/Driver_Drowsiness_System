"""
balance_dataset.py — Trim & Augment to Perfect Balance
Driver Drowsiness Detection System

Does two things in one run:
  1. Trims open_eye/ down to TARGET (deletes random excess files)
  2. Augments yawn/ up to TARGET (generates new images via transforms)

Target: 90,000 per class → 270,000 total balanced dataset

Usage:
    python src/balance_dataset.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_PATH   = r"D:\Projects\Driver_Drowsiness_System\dataset\train"

CLOSED_EYE   = os.path.join(TRAIN_PATH, "closed_eye")
OPEN_EYE     = os.path.join(TRAIN_PATH, "open_eye")
YAWN         = os.path.join(TRAIN_PATH, "yawn")

TARGET       = 90_000   # target per class

SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────
# AUGMENTATION TRANSFORMS
# ──────────────────────────────────────────────
def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def random_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

def random_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))

def random_rotation(img):
    return img.rotate(random.uniform(-25, 25), resample=Image.BILINEAR, expand=False)

def random_zoom(img):
    w, h   = img.size
    scale  = random.uniform(0.78, 0.95)
    nw, nh = int(w * scale), int(h * scale)
    left   = random.randint(0, w - nw)
    top    = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

def add_noise(img):
    arr   = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def random_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

def color_jitter(img):
    img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))
    return img

TRANSFORMS = [
    horizontal_flip,
    random_brightness,
    random_contrast,
    random_rotation,
    random_zoom,
    add_noise,
    random_blur,
    color_jitter,
]

def augment_image(img: Image.Image) -> Image.Image:
    """Apply 2–4 random transforms."""
    for fn in random.sample(TRANSFORMS, k=random.randint(2, 4)):
        img = fn(img)
    return img


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def count_images(folder: str) -> int:
    return len([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ])

def list_images(folder: str) -> list:
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]

def progress_bar(done: int, total: int, width: int = 28) -> str:
    pct  = done / total
    fill = int(pct * width)
    bar  = "█" * fill + "░" * (width - fill)
    return f"[{bar}] {pct*100:5.1f}%  {done:,}/{total:,}"


# ──────────────────────────────────────────────
# STEP 1 — TRIM open_eye
# ──────────────────────────────────────────────
def trim_folder(folder: str, target: int, label: str):
    files   = list_images(folder)
    current = len(files)

    if current <= target:
        print(f"  ✅ {label} already at {current:,} — no trimming needed.\n")
        return

    to_delete = current - target
    print(f"  Trimming {label}: {current:,} → {target:,}  (deleting {to_delete:,} files)")

    victims = random.sample(files, to_delete)
    for i, fname in enumerate(victims, 1):
        os.remove(os.path.join(folder, fname))
        if i % 5000 == 0 or i == to_delete:
            print(f"    {progress_bar(i, to_delete)}")

    print(f"  ✅ {label} trimmed → {count_images(folder):,} files\n")


# ──────────────────────────────────────────────
# STEP 2 — AUGMENT yawn
# ──────────────────────────────────────────────
def augment_folder(folder: str, target: int, label: str):
    files   = list_images(folder)
    current = len(files)

    if current >= target:
        print(f"  ✅ {label} already at {current:,} — no augmentation needed.\n")
        return

    needed = target - current
    print(f"  Augmenting {label}: {current:,} → {target:,}  (generating {needed:,} images)")

    generated = 0
    while generated < needed:
        src_name = random.choice(files)
        src_path = os.path.join(folder, src_name)

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception:
            continue

        aug   = augment_image(img)
        stem, ext = os.path.splitext(src_name)
        out_name  = f"{stem}_aug_{generated:06d}{ext}"
        out_path  = os.path.join(folder, out_name)
        aug.save(out_path, quality=92)

        generated += 1

        if generated % 2000 == 0 or generated == needed:
            print(f"    {progress_bar(generated, needed)}")

    print(f"  ✅ {label} augmented → {count_images(folder):,} files\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'='*58}")
    print(f"  Dataset Balance Script")
    print(f"  Target per class : {TARGET:,}")
    print(f"{'='*58}\n")

    # ── Audit before ───────────────────────────
    ce = count_images(CLOSED_EYE)
    oe = count_images(OPEN_EYE)
    ya = count_images(YAWN)

    print(f"  BEFORE:")
    print(f"    closed_eye → {ce:,}")
    print(f"    open_eye   → {oe:,}")
    print(f"    yawn       → {ya:,}")
    print(f"    total      → {ce+oe+ya:,}\n")

    # ── closed_eye: already ~91k, trim slightly ─
    print(f"── Step 1: closed_eye ─────────────────────")
    trim_folder(CLOSED_EYE, TARGET, "closed_eye")

    # ── open_eye: trim from 127k to 90k ────────
    print(f"── Step 2: open_eye ───────────────────────")
    trim_folder(OPEN_EYE, TARGET, "open_eye")

    # ── yawn: augment from 38k to 90k ──────────
    print(f"── Step 3: yawn ───────────────────────────")
    augment_folder(YAWN, TARGET, "yawn")

    # ── Audit after ────────────────────────────
    ce = count_images(CLOSED_EYE)
    oe = count_images(OPEN_EYE)
    ya = count_images(YAWN)

    print(f"\n{'='*58}")
    print(f"  AFTER:")
    print(f"    closed_eye → {ce:,}")
    print(f"    open_eye   → {oe:,}")
    print(f"    yawn       → {ya:,}")
    print(f"    total      → {ce+oe+ya:,}")

    # Balance check
    counts  = [ce, oe, ya]
    ratio   = min(counts) / max(counts)
    print(f"\n  Balance ratio : {ratio:.3f}  {'✅ Perfect' if ratio >= 0.97 else '⚠ Check manually'}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()