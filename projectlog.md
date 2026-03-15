# Driver Drowsiness Detection System — Project Log

## Project Overview
A real-time driver drowsiness detection system that detects:
- **Eye closure** (EAR — Eye Aspect Ratio)
- **Yawning** (MAR — Mouth Aspect Ratio)
- **Head tilt / nodding** (head pose estimation)

Triggers an alarm when drowsiness is detected and logs all events to SQLite.

**Designed to run in two modes:**
- `virtual` — webcam / any USB camera (laptop, desktop)
- `hardware` — physical sensors (Raspberry Pi camera, IR sensor, ultrasonic, buzzer)

Both modes share the same CNN model, database, and alert logic. Only the input/output layer changes.

---

## Stack
| Component | Library |
|---|---|
| Camera & frame capture | OpenCV |
| Face landmarks (468 points) | MediaPipe |
| CNN model | TensorFlow + Keras |
| Model evaluation | scikit-learn |
| Data processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Alert sound | Pygame |
| Database | SQLite (built-in) |
| Hardware mode | RPi.GPIO / picamera2 (optional) |

---

## Project Structure
```
Driver_Drowsiness_System/
├── .gitignore
├── projectlog.md
├── requirements.txt
├── README.md
├── drowsiness.db                  # SQLite database (auto-created)
│
├── src/
│   ├── db_setup.py                # Day 1 ✅
│   ├── split_dataset.py           # Day 1 ✅
│   ├── balance_dataset.py         # Day 1 ✅
│   ├── extract_yawdd_frames_v2.py # Day 1 ✅
│   ├── inspect_avi_dataset.py     # Day 1 ✅
│   └── augment_yawn.py            # Day 1 ✅
│
├── core/                          # Day 3+
│   ├── detector.py                # EAR / MAR / head pose logic
│   ├── alert_manager.py           # alarm + DB logging
│   └── session_manager.py         # session start/end
│
├── database/
│   └── db_queries.py              # CRUD helpers (Day 3+)
│
├── models/                        # Day 2+
│   └── drowsiness_model.keras     # trained CNN weights
│
├── data/
│   └── alarm.wav                  # alert sound file
│
└── dataset/                       # gitignored — local only
    ├── train/      → 216,000 images
    ├── validation/ →  27,000 images
    └── test/       →  27,000 images
```

---

## Mode Architecture

### Virtual Mode (webcam)
```
Webcam (OpenCV)
    → MediaPipe Face Mesh (468 landmarks)
    → EAR / MAR / Head Pose calculation
    → CNN model prediction
    → Alert Manager (pygame alarm)
    → SQLite logging
```

### Hardware Mode (Raspberry Pi / sensors)
```
Pi Camera / IR Camera
    → MediaPipe Face Mesh
    → EAR / MAR / Head Pose calculation
    → CNN model prediction
    → Alert Manager (GPIO buzzer + LED)
    → SQLite logging
```

**Key design decision:** Both modes use identical `core/` logic.
Only `main.py` switches between `mode = "virtual"` and `mode = "hardware"`.

---

## Day Log

---

### Day 1 — Database + Dataset Setup
**Date:** 2026-03-15
**Status:** ✅ Complete

#### What was done
- SQLite database created with full schema
  - Tables: drivers, sessions, alerts, frame_logs, config, migrations
  - 7 default config values seeded (EAR threshold, MAR threshold, cooldown, etc.)
  - Migration system in place for future schema changes
- YawDD dataset (AVI videos) extracted and processed
  - Source: YawDD Mirror folder (322 videos, Male + Female)
  - Extracted frames at FRAME_STEP=2 using OpenCV
  - Label detection from filename (-Normal → open_eye, -Yawning → yawn)
  - Talking videos skipped (ambiguous label)
- Dataset balanced to 90,000 per class
  - closed_eye: 91,566 → 90,000 (trimmed)
  - open_eye: 127,832 → 90,000 (trimmed, removed v1+v2 duplicates)
  - yawn: 38,988 → 90,000 (augmented — flip, brightness, rotation, zoom, noise)
- Train / Val / Test split completed
  - 80% train / 10% validation / 10% test
  - Perfectly stratified — 72k/9k/9k per class

#### Final dataset numbers
| Split | closed_eye | open_eye | yawn | Total |
|---|---|---|---|---|
| train | 72,000 | 72,000 | 72,000 | 216,000 |
| validation | 9,000 | 9,000 | 9,000 | 27,000 |
| test | 9,000 | 9,000 | 9,000 | 27,000 |
| **total** | **90,000** | **90,000** | **90,000** | **270,000** |

#### Database state
| Table | Rows |
|---|---|
| drivers | 1 (Test Driver) |
| sessions | 0 |
| alerts | 0 |
| frame_logs | 0 |
| config | 7 |
| migrations | 1 |

#### Scripts created
| Script | Purpose |
|---|---|
| `src/db_setup.py` | Creates SQLite DB, tables, indexes, seed data |
| `src/extract_yawdd_frames_v2.py` | Extracts frames from YawDD AVI zip |
| `src/balance_dataset.py` | Trims open_eye + augments yawn to 90k each |
| `src/split_dataset.py` | 80/10/10 train/val/test split |

#### Issues faced & resolved
| Issue | Fix |
|---|---|
| pdfplumber _cffi_backend error | Manually extracted PDFs from zip |
| Mirror filenames had no action label in Dash videos | Used Mirror folder only (clean labels) |
| FRAME_STEP=5 gave only 13k frames | Reduced to FRAME_STEP=2, then audited counts |
| v1+v2 extraction caused open_eye duplicates | Trimmed to 90k via balance_dataset.py |
| yawn only 38k vs 90k target | Augmented with 8 transforms to reach 90k |

---

### Day 2 — CNN Model
**Date:** _(pending)_
**Status:** 🔄 Upcoming

#### Plan
- [ ] Build CNN using MobileNetV2 (transfer learning)
- [ ] ImageDataGenerator with augmentation for training
- [ ] Train on 216,000 images
- [ ] Evaluate on validation set
- [ ] Plot accuracy + loss curves
- [ ] Save best model as `models/drowsiness_model.keras`
- [ ] Test on 27,000 test images
- [ ] Generate confusion matrix per class

#### Target metrics
| Metric | Target |
|---|---|
| Training accuracy | > 92% |
| Validation accuracy | > 90% |
| Yawn class accuracy | > 85% |

---

### Day 3 — Real-time Detection (planned)
**Status:** 📋 Planned

- MediaPipe face mesh → EAR + MAR calculation
- Load trained CNN model
- Real-time webcam loop
- Alert manager (pygame alarm)
- SQLite session + alert logging
- Virtual mode complete

---

### Day 4 — Hardware Mode (planned)
**Status:** 📋 Planned

- Raspberry Pi camera integration
- GPIO buzzer + LED alerts
- Same core/ logic as virtual mode
- Test on Pi hardware

---

### Day 5 — Dashboard (planned)
**Status:** 📋 Planned

- Session history viewer
- EAR/MAR charts per session
- Alert frequency analysis
- Export session report

---

## Key Decisions Log

| Decision | Reason |
|---|---|
| MediaPipe over dlib | No .dat file needed, 468 landmarks vs 68, easier install |
| SQLite over PostgreSQL | Lightweight, no server needed, works on Pi and laptop |
| MobileNetV2 for CNN | Fast inference, small model size, good accuracy, works on Pi |
| YawDD Mirror over Dash | Mirror has clean per-action labels; Dash has mixed behavior |
| 90k per class target | Enough for strong CNN, avoids overfitting, balanced |
| Dual mode design | Same codebase runs on webcam and Raspberry Pi hardware |

---

## Hardware Mode Notes (future reference)
When switching to hardware mode:
- Replace `cv2.VideoCapture(0)` with `picamera2` stream
- Replace `pygame.mixer` alarm with `RPi.GPIO` buzzer on pin X
- Add LED indicator on GPIO pin Y
- SQLite DB path changes to `/home/pi/drowsiness.db`
- All `core/` files remain identical — zero changes needed

---

*Log maintained by: Austin Trinidad*
*Last updated: Day 1 — 2026-03-15*