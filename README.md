# Driver Drowsiness Detection System 🛣️😴

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-green)](https://tensorflow.org/)

Real-time **driver drowsiness detection** using:
- **MediaPipe Face Mesh** (468 landmarks for EAR/MAR)
- **CNN** (MobileNetV2, **96.19% test accuracy** on 27k images)
- **Multi-modal fusion** (EAR + CNN agreement)
- **Audio + visual alerts** (beep + OpenCV overlay)
- **SQLite logging** (sessions, alerts, frame metrics)

**Virtual mode** (webcam) fully working. **Hardware mode** (RPi stub) ready for Day 4.

Watch it [detect & alert](https://github.com/user/repo/blob/main/projectlog.md#day-3-real-time-detection-virtual-mode)!

## 🚀 Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd Driver_Drowsiness_System

# 2. Virtual Environment (recommended)
python -m venv venv
# Windows:
venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Setup Database (creates drowsiness.db)
python src/db_setup.py

# 5. Run (auto-detects webcam)
python main.py

# Controls:
# - Press 'Q' to quit
# - Webcam overlay shows: EAR, MAR, CNN class/conf, drowsy frame counter
```

**Expected Output:**
```
[DETECTOR] Loading CNN model: models/drowsiness_model.keras
[DETECTOR] CNN model loaded and warmed up.
[DETECTOR] MediaPipe face mesh ready.
[CAM] Camera 0 found — 640x480
[DB] Session started — id: 1
[LOOP] Detection running. Press Q in the window to stop.
```

## 📋 Detailed Setup

### Prerequisites
- Python 3.10+
- Webcam (built-in or USB)
- GPU optional (RTX/CUDA speeds training/inference)

### 1. Git Clone
```bash
git clone https://github.com/yourusername/Driver_Drowsiness_System.git
cd Driver_Drowsiness_System
```
Models use Git LFS — `git lfs pull` if prompted.

### 2. Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Dependencies
```bash
pip install -r requirements.txt
```
**Notes:**
- ~2GB install (TensorFlow, OpenCV, MediaPipe).
- **Protobuf conflict?** `pip install protobuf==3.20.3`
- **Matplotlib issues?** `pip install matplotlib==3.9.2`

### 4. Database
```bash
python src/db_setup.py
```
Creates `drowsiness.db` with tables: `drivers`, `sessions`, `alerts`, `frame_logs`, `config`.

### 5. Verify Models
```
models/
├── drowsiness_model.keras  # Production model (96.19% acc)
└── phase1_best.keras      # Training checkpoint
```
If missing: Run `python src/train_model.py`.

### 6. Run
```bash
# Virtual mode (default)
python main.py

# Force camera index
python main.py --cam 1

# Hardware mode (Day 4 stub)
python main.py --mode hardware
```

## 🎛️ Usage

### Modes
| Mode | Command | Input | Output |
|------|---------|-------|--------|
| **Virtual** | `python main.py` | Webcam (auto 0/1/2) | OpenCV window + beep |
| **Hardware** | `python main.py --mode hardware` | RPi Camera/GPIO (TBD) | Buzzer/LED |

### Overlay Elements
- **EAR** (Eye Aspect Ratio): <0.20 → drowsy eyes
- **MAR** (Mouth Aspect Ratio): >0.60 → yawn
- **CNN**: Class (closed/open/yawn) + confidence bar
- **Frame Counter**: Progress to alert (30 frames ~1s @30FPS)
- **Status**: OK/DROWSY/ALERT/NO_FACE

### Config (in DB)
| Key | Default | Description |
|-----|---------|-------------|
| `ear_threshold` | 0.20 | EAR below → eye closure |
| `cnn_threshold` | 0.85 | CNN confidence min |
| `frame_threshold` | 30 | Consecutive drowsy frames → alert |

## 🗂️ Project Structure

```
Driver_Drowsiness_System/
├── README.md                 # 👈 You are here
├── main.py                   # Entry point
├── requirements.txt          # ML deps
├── projectlog.md             # Detailed dev log
├── TODO.md                   # Current tasks
├── drowsiness.db             # SQLite (gitignored)
│
├── core/                     # Detection engine
│   ├── detector.py           # EAR/MAR/CNN fusion
│   ├── alert_manager.py      # Beep + overlay
│   └── session_manager.py    # Session logging
│
├── database/
│   └── db_queries.py         # SQLite CRUD
│
├── src/                      # Data/Training
│   ├── db_setup.py           # DB init
│   ├── train_model.py        # MobileNetV2 (96% acc)
│   └── extract_yawdd...      # Dataset prep
│
├── models/                   # Git LFS
│   └── drowsiness_model.keras
├── dataset/                  # Gitignored (270k images)
└── logs/                     # Training plots/logs
```

## 🔧 Training Your Own Model

1. Prepare YawDD dataset (see `src/extract_yawdd_frames_v2.py`)
2. Balance/split: `python src/balance_dataset.py`
3. Train:
   ```bash
   python src/train_model.py
   ```
   - 2-phase: Frozen base → Fine-tune top layers
   - Mixed precision (RTX friendly)
   - Checkpoints saved to `models/`

**Test Results:** 96.19% acc, 97% yawn recall (27k test images)

## 🚨 Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'mediapipe'` | `pip install mediapipe==0.10.32` |
| `protobuf` conflict | `pip install protobuf==3.20.3` |
| `No camera found` | Check USB/webcam, try `--cam 1` |
| `Model not found` | `python src/train_model.py` |
| **Windows FPS low** | Uses `CAP_DSHOW` — unplug/replug camera |
| Alerts too sensitive | Edit `core/detector.py` thresholds |
| DB errors | Delete `drowsiness.db`, rerun `db_setup.py` |

## 📈 Performance
- **FPS**: 25-30 on i5/RTX2050 webcam
- **Model Size**: ~10MB (MobileNetV2 quantized)
- **Inference**: <20ms/frame end-to-end
- **Alerts**: 0 false positives after tuning

## 🤝 Contributing
1. Fork & PR
2. Update `projectlog.md` with changes
3. `black .` for formatting
4. Tests: Add to `tests/` (TBD)

## 📄 License
MIT — Free for commercial use.

## 🙏 Credits
- **YawDD Dataset** for training data
- Built over 3 days by Austin Trinidad
- See full [projectlog.md](./projectlog.md) for dev journey

**Hardware mode (RPi + buzzer) coming Day 4!** 🚀

---

*Last updated: Post-Day 3*

