import zipfile
ZIP_PATH = r"D:\Projects\Driver_Drowsiness_System\dataset\archive.zip"

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    mirror_files = [n for n in z.namelist() if "Mirror" in n and n.endswith(".avi")]
    for f in mirror_files[:25]:
        print(f)