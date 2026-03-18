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
│   ├── augment_yawn.py            # Day 1 ✅
│   ├── train_model.py             # Day 2 ✅
│   └── evaluate_model.py          # Day 2 ✅
│
├── core/                          # Day 3+
│   ├── detector.py                # EAR / MAR / head pose logic
│   ├── alert_manager.py           # alarm + DB logging
│   └── session_manager.py         # session start/end
│
├── database/
│   └── db_queries.py              # CRUD helpers (Day 3+)
│
├── models/
│   ├── drowsiness_model.keras     # Day 2 ✅ final trained model (96.19% test acc)
│   └── phase1_best.keras          # Day 2 ✅ Phase 1 checkpoint (86.99% val acc)
│
├── logs/
│   ├── training/                  # TensorBoard logs + CSV training log
│   └── plots/
│       ├── training_curves.png    # Day 2 ✅ accuracy + loss curves
│       └── confusion_matrix.png   # Day 2 ✅ per-class confusion matrix
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

### Day 2 — CNN Model Training & Evaluation
**Date:** 2026-03-18
**Status:** ✅ Complete

#### What was done
- MobileNetV2 CNN built using transfer learning (pretrained ImageNet weights)
- Two-phase training strategy implemented
  - Phase 1: frozen MobileNetV2 base, trained classification head only (10 epochs)
  - Phase 2: unfroze top 36 layers (from index 100), fine-tuned with LR=1e-5
- Mixed precision (float16) enabled for 4 GB VRAM compatibility on RTX 2050
- GPU memory growth enabled to prevent OOM errors
- Heavy augmentation on training generator (rotation, flip, brightness, zoom, shear)
- Class weights computed as safety net for any residual imbalance
- Model evaluated on 27,000 held-out test images
- Confusion matrix and training curves saved to `logs/plots/`

#### Model architecture
```
Input (96x96x3)
    → MobileNetV2 backbone (pretrained ImageNet, top 36 layers unfrozen)
    → GlobalAveragePooling2D
    → Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(3, softmax)   [float32 cast for mixed precision stability]
```

#### Training progression
| Phase | Epochs | Best Val Accuracy | Notes |
|---|---|---|---|
| Phase 1 (frozen base) | 10 | 86.99% | Head-only training |
| Phase 2 (fine-tuning) | 4 of 25 | **93.05%** | Interrupted, resumed from checkpoint |
| Phase 2 resume | +4 epochs | **96.19% (test)** | EarlyStopping triggered |

#### Final test set results (27,000 images)
| Metric | Result | Target | Status |
|---|---|---|---|
| Overall accuracy | **96.19%** | >90% | ✅ PASS |
| Yawn recall | **97.32%** | >85% | ✅ PASS |
| closed_eye F1 | 96.57% | — | ✅ |
| open_eye F1 | 94.23% | — | ✅ |
| yawn F1 | 97.76% | — | ✅ |

#### Per-class results
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| closed_eye | 95.08% | 98.10% | 96.57% | 9,000 |
| open_eye | 95.33% | 93.16% | 94.23% | 9,000 |
| yawn | 98.21% | 97.32% | 97.76% | 9,000 |

#### Confusion matrix (raw counts)
| Actual \ Predicted | closed_eye | open_eye | yawn |
|---|---|---|---|
| closed_eye | **8,829** | 170 | 1 |
| open_eye | 457 | **8,384** | 159 |
| yawn | 0 | 241 | **8,759** |

#### Confidence analysis
| Class | Mean confidence | Min | Max |
|---|---|---|---|
| closed_eye | 96.05% | 2.20% | 100% |
| open_eye | 91.87% | 0.01% | 100% |
| yawn | 96.62% | 0.19% | 100% |

Low-confidence predictions (<70%): 1,023 / 27,000 (3.8%)
These will trigger uncertainty handling in Day 3 real-time detection.

#### Scripts created
| Script | Purpose |
|---|---|
| `src/train_model.py` | Two-phase MobileNetV2 training with GPU optimizations |
| `src/evaluate_model.py` | Test set evaluation, confusion matrix, confidence analysis |

#### Issues faced & resolved
| Issue | Fix |
|---|---|
| matplotlib 3.10.x broken install (_c_internal_utils error) | Downgraded to matplotlib==3.9.2 |
| CosineDecay schedule incompatible with TF 2.15 Adam | Replaced with plain float LR (1e-5), ReduceLROnPlateau handles decay |
| VS Code closed mid-epoch 5, interrupted Phase 2 | Epoch 4 checkpoint (93.05%) was already saved by ModelCheckpoint — no data lost. Resumed from checkpoint. |

#### Key observations
- Yawn class achieved highest accuracy (97.76% F1) despite starting with only 38k raw images — Day 1 augmentation was highly effective
- Main confusion is open_eye ↔ closed_eye (457 misclassifications) — expected for squinting/partial blink edge cases. Will not affect real-time system since alarms require multiple consecutive frames
- Zero yawn frames misclassified as closed_eye — critical for safety (no false drowsiness alarms from yawning)
- 3.8% low-confidence frames will be handled gracefully in Day 3 detector logic

---

### Day 3 — Real-time Detection (planned)
**Status:** 📋 Planned

- MediaPipe face mesh → EAR + MAR calculation
- Load trained CNN model (`drowsiness_model.keras`)
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
| img_size=96x96, batch=16 | Fits within 4 GB VRAM with mixed precision on RTX 2050 |
| Two-phase training | Phase 1 warms up head fast; Phase 2 fine-tunes for max accuracy |
| Mixed precision (float16) | Reduces VRAM ~40%, speeds up RTX training with no accuracy loss |
| Unfreeze from layer 100 | Top 36 MobileNetV2 layers fine-tuned; lower layers keep ImageNet features |

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
*Last updated: Day 2 — 2026-03-18*