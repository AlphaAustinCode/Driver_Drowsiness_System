import os

TRAIN_PATH = r"D:\Projects\Driver_Drowsiness_System\dataset\train"

print(f"\n── Train Folder Audit ──\n")
total = 0
for cls in os.listdir(TRAIN_PATH):
    cls_path = os.path.join(TRAIN_PATH, cls)
    if os.path.isdir(cls_path):
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        print(f"  {cls:<20} → {len(files):>8,} files")
        total += len(files)

print(f"\n  {'TOTAL':<20} → {total:>8,} files")