"""
extract_yawdd_frames_v2.py — YawDD Mirror Frame Extractor (Fixed)
Driver Drowsiness Detection System

Changes from v1:
  - FRAME_STEP reduced to 2 (extracts 15 frames/sec instead of 6)
  - Explicit Mirror path filter (no Dash bleed-through)
  - Appends to existing frames (won't re-extract already saved ones)
  - Shows per-class running total while extracting

Usage:
    python src/extract_yawdd_frames_v2.py
"""

import os
import cv2
import zipfile
import tempfile
import shutil

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ZIP_PATH     = r"D:\Projects\Driver_Drowsiness_System\dataset\archive.zip"  # ← update

DATASET_PATH = r"D:\Projects\Driver_Drowsiness_System\dataset"
YAWN_OUT     = os.path.join(DATASET_PATH, "train", "yawn")
OPEN_EYE_OUT = os.path.join(DATASET_PATH, "train", "open_eye")

# FRAME_STEP = 2 → 15 frames/sec from 30fps
# Gives ~3x more frames than v1 (step=5)
FRAME_STEP = 2

# Only process Mirror folder paths
MIRROR_PATHS = [
    "Mirror/Mirror/Female_mirror/",
    "Mirror/Mirror/Male_mirror",     # has a space in original zip
]

MIN_FRAMES = 30   # skip corrupt/tiny videos


# ──────────────────────────────────────────────
# LABEL DETECTION
# ──────────────────────────────────────────────
def get_label(filename: str):
    """
    Detects label from Mirror filenames like:
      1-FemaleNoGlasses-Normal.avi
      10-MaleGlasses-Yawning.avi
      15-FemaleSunGlasses-Normal.avi

    Returns 'yawn', 'open_eye', or None (skip Talking)
    """
    name = filename.lower().strip()

    # Fix double extension bug
    if name.endswith(".avi.avi"):
        name = name[:-4]
    name = name.replace(".avi", "")

    # Action is always the LAST segment after final hyphen
    parts = name.split("-")
    action = parts[-1].strip() if parts else ""

    if action == "yawning":
        return "yawn"
    elif action == "normal":
        return "open_eye"
    else:
        return None   # talking, mixed, or unknown → skip


# ──────────────────────────────────────────────
# FRAME EXTRACTOR
# ──────────────────────────────────────────────
def extract_frames(video_path: str, out_dir: str, video_id: str, frame_step: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < MIN_FRAMES:
        cap.release()
        return 0

    os.makedirs(out_dir, exist_ok=True)
    saved      = 0
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            fname = f"{video_id}_f{frame_idx:06d}.jpg"
            fpath = os.path.join(out_dir, fname)

            # Skip if already extracted (safe to re-run)
            if not os.path.exists(fpath):
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def is_mirror(entry: str) -> bool:
    return any(mp in entry for mp in MIRROR_PATHS)

def count_existing(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'='*62}")
    print(f"  YawDD Frame Extractor v2")
    print(f"  ZIP        : {ZIP_PATH}")
    print(f"  Frame step : every {FRAME_STEP} frames (~{30//FRAME_STEP} fps)")
    print(f"{'='*62}")

    # Show existing counts
    existing_yawn     = count_existing(YAWN_OUT)
    existing_open_eye = count_existing(OPEN_EYE_OUT)
    print(f"\n  Existing frames:")
    print(f"    yawn/     → {existing_yawn:,} (will append new frames)")
    print(f"    open_eye/ → {existing_open_eye:,} (will append new frames)\n")

    stats = {
        "yawn"    : {"videos": 0, "frames": 0},
        "open_eye": {"videos": 0, "frames": 0},
        "skipped" : 0,
        "errors"  : 0,
    }

    tmp_dir = tempfile.mkdtemp(prefix="yawdd_v2_")

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            all_entries = z.namelist()

            # Mirror AVI files only
            avi_entries = [
                n for n in all_entries
                if is_mirror(n)
                and (".avi" in n.lower())
            ]

            print(f"  Mirror AVI files found: {len(avi_entries)}\n")

            for i, entry in enumerate(avi_entries):
                filename = entry.split("/")[-1]
                label    = get_label(filename)

                if label is None:
                    stats["skipped"] += 1
                    print(f"  [{i+1:>3}/{len(avi_entries)}] SKIP     {filename}")
                    continue

                out_dir  = YAWN_OUT if label == "yawn" else OPEN_EYE_OUT
                video_id = (
                    filename
                    .replace(".avi.avi", "")
                    .replace(".avi", "")
                    .replace(" ", "_")
                    .strip()
                )

                # Extract video to temp
                tmp_video = os.path.join(tmp_dir, "current.avi")
                try:
                    with z.open(entry) as src, open(tmp_video, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except Exception as e:
                    print(f"  [{i+1:>3}/{len(avi_entries)}] ERROR    {filename}: {e}")
                    stats["errors"] += 1
                    continue

                saved = extract_frames(tmp_video, out_dir, video_id, FRAME_STEP)

                tag = "YAWN    " if label == "yawn" else "OPEN_EYE"
                stats[label]["videos"] += 1
                stats[label]["frames"] += saved

                # Running totals
                total_yawn     = existing_yawn     + stats["yawn"]["frames"]
                total_open_eye = existing_open_eye + stats["open_eye"]["frames"]

                print(
                    f"  [{i+1:>3}/{len(avi_entries)}] {tag} "
                    f"{filename:<42} +{saved:>5} frames  "
                    f"[yawn:{total_yawn:,}  open:{total_open_eye:,}]"
                )

                if os.path.exists(tmp_video):
                    os.remove(tmp_video)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── SUMMARY ────────────────────────────────
    final_yawn     = existing_yawn     + stats["yawn"]["frames"]
    final_open_eye = existing_open_eye + stats["open_eye"]["frames"]

    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"{'='*62}")
    print(f"  yawn/      → {stats['yawn']['videos']:>3} videos  +{stats['yawn']['frames']:>7,} frames  total: {final_yawn:,}")
    print(f"  open_eye/  → {stats['open_eye']['videos']:>3} videos  +{stats['open_eye']['frames']:>7,} frames  total: {final_open_eye:,}")
    print(f"  skipped    → {stats['skipped']:>3} (Talking videos)")
    print(f"  errors     → {stats['errors']:>3}")

    ratio = min(final_yawn, final_open_eye) / max(final_yawn, final_open_eye) if max(final_yawn, final_open_eye) > 0 else 0
    print(f"\n  Balance ratio: {ratio:.2f}  {'✅ Good' if ratio >= 0.8 else '⚠ Consider augmenting smaller class'}")

    if final_yawn < 30000 or final_open_eye < 30000:
        print(f"\n  ⚠  Frame count still low.")
        print(f"     Try FRAME_STEP = 1 to extract every frame.")
    else:
        print(f"\n  ✅ Frame count looks good for CNN training!")
    print()


if __name__ == "__main__":
    main()