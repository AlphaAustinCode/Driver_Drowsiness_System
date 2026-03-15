import os
import cv2

DATASET_PATH = "D:\Projects\Driver_Drowsiness_System\dataset"
IMG_SIZE = 64
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def clean_and_resize(folder: str, img_size: int = IMG_SIZE) -> None:
    removed = 0
    processed = 0
    skipped = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext not in VALID_EXTENSIONS:
                skipped += 1
                continue

            try:
                img = cv2.imread(path)

                if img is None:
                    print(f"  [REMOVED] Unreadable: {path}")
                    os.remove(path)
                    removed += 1
                    continue

                resized = cv2.resize(img, (img_size, img_size),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, resized)
                processed += 1

            except Exception as e:
                print(f"  [REMOVED] Error ({e}): {path}")
                os.remove(path)
                removed += 1

    print(f"\nDone — Processed: {processed} | Removed: {removed} | Skipped: {skipped}")

clean_and_resize(DATASET_PATH)