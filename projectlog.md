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
| Alert sound | Pygame (synthesized beep — no wav file needed) |
| Database | SQLite (built-in) |
| Hardware mode | RPi.GPIO / picamera2 (Day 4) |

---

## Project Structure
```
Driver_Drowsiness_System/
├── .gitignore
├── projectlog.md
├── requirements.txt
├── README.md
├── main.py                        # Day 3 ✅ entry point, virtual/hardware mode switch
├── drowsiness.db                  # SQLite database (auto-created)
│
├── core/                          # Day 3 ✅
│   ├── __init__.py
│   ├── detector.py                # EAR / MAR / CNN / fusion logic
│   ├── alert_manager.py           # beep + OpenCV overlay + DB logging
│   └── session_manager.py         # session start/end/frame logging
│
├── database/                      # Day 3 ✅
│   ├── __init__.py
│   └── db_queries.py              # CRUD helpers for SQLite
│
├── src/
│   ├── db_setup.py                # Day 1 ✅ (updated Day 3)
│   ├── split_dataset.py           # Day 1 ✅
│   ├── balance_dataset.py         # Day 1 ✅
│   ├── extract_yawdd_frames_v2.py # Day 1 ✅
│   ├── inspect_avi_dataset.py     # Day 1 ✅
│   ├── augment_yawn.py            # Day 1 ✅
│   ├── train_model.py             # Day 2 ✅
│   └── evaluate_model.py          # Day 2 ✅
│
├── models/
│   ├── drowsiness_model.keras     # Day 2 ✅ final trained model (96.19% test acc)
│   └── phase1_best.keras          # Day 2 ✅ Phase 1 checkpoint
│
├── logs/
│   ├── training/                  # TensorBoard logs + CSV training log
│   └── plots/
│       ├── training_curves.png    # Day 2 ✅
│       └── confusion_matrix.png   # Day 2 ✅
│
├── data/
│   └── alarm.wav                  # optional — system uses synthesized beep
│
└── dataset/                       # gitignored — local only
    ├── train/      → 216,000 images
    ├── validation/ →  27,000 images
    └── test/       →  27,000 images
```

---

## Mode Architecture

### Virtual Mode (webcam) — Day 3 ✅
```
Webcam (OpenCV, auto-detect)
    → MediaPipe Face Mesh (468 landmarks)
    → EAR / MAR calculation
    → CNN model inference (direct call, no data pipeline)
    → Fusion logic (EAR + CNN combined decision)
    → Frame counter (30 consecutive drowsy frames → alarm)
    → Alert Manager (synthesized beep + OpenCV overlay)
    → SQLite logging (sessions, alerts, frame_logs)
```

### Hardware Mode (Raspberry Pi) — Day 4 planned
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

#### Final test set results (27,000 images)
| Metric | Result | Target | Status |
|---|---|---|---|
| Overall accuracy | **96.19%** | >90% | ✅ PASS |
| Yawn recall | **97.32%** | >85% | ✅ PASS |
| closed_eye F1 | 96.57% | — | ✅ |
| open_eye F1 | 94.23% | — | ✅ |
| yawn F1 | 97.76% | — | ✅ |

#### Scripts created
| Script | Purpose |
|---|---|
| `src/train_model.py` | Two-phase MobileNetV2 training with GPU optimizations |
| `src/evaluate_model.py` | Test set evaluation, confusion matrix, confidence analysis |

#### Issues faced & resolved
| Issue | Fix |
|---|---|
| matplotlib 3.10.x broken install | Downgraded to matplotlib==3.9.2 |
| CosineDecay incompatible with TF 2.15 Adam | Replaced with plain float LR, ReduceLROnPlateau handles decay |
| VS Code closed mid-epoch 5 | Epoch 4 checkpoint (93.05%) already saved — resumed from there |

---

### Day 3 — Real-time Detection (Virtual Mode)
**Date:** 2026-03-19
**Status:** ✅ Complete

#### What was done
- Full real-time detection pipeline built and tested on laptop webcam
- MediaPipe FaceMesh (468 landmarks) for EAR + MAR calculation
- CNN inference integrated using direct model call (`model(img, training=False)`) for real-time performance
- Fusion logic combining EAR and CNN — requires both to agree on eye closure (eliminates false positives)
- Frame counter with gradual decay — 30 consecutive drowsy frames (~1 sec) before alarm fires
- Synthesized beep alert generated in code using numpy + pygame (no wav file needed)
- OpenCV overlay showing EAR, MAR, CNN class + confidence bar, drowsy frame progress bar
- Auto camera detection — tries indices 0, 1, 2 automatically
- Full SQLite logging — sessions, per-alert records, per-frame metrics (throttled every 15 frames)
- DB schema updated to match Day 3 column names (start_time, end_time, timestamp, cnn_class, cnn_confidence)

#### Detection architecture (per frame)
```
Webcam frame
    → MediaPipe → 468 landmarks
    → EAR (avg left + right eye)
    → MAR (mouth open ratio)
    → CNN face crop → model(img) → [closed_eye, open_eye, yawn] probabilities
    → Fusion: (CNN closed_eye AND EAR < 0.20) → combined alert
              (EAR < 0.20 alone)              → eye closure alert
              (CNN yawn OR MAR > 0.60)        → yawn alert
    → Frame counter (30 frames sustained) → trigger alarm
    → AlertManager: beep + overlay + DB log
```

#### Final calibrated thresholds
| Parameter | Initial | Final | Reason |
|---|---|---|---|
| `EAR_THRESHOLD` | 0.25 | 0.20 | Natural open-eye EAR was ~0.30+ |
| `CNN_THRESHOLD` | 0.70 | 0.85 | Reduce CNN false positives |
| `FRAME_THRESHOLD` | 20 | 30 | ~1 sec sustained closure needed |
| Frame counter decay | -1/frame | -3/frame | Fast reset after blinks |
| CNN-alone eye rule | enabled | removed | EAR is ground truth for eyes |

#### Test results
| Session | Frames | Eyes open alerts | Eyes closed alerts | Result |
|---|---|---|---|---|
| Initial (no tuning) | 155 | 17 false alerts | — | ❌ Too sensitive |
| After EAR/CNN fix | 499 | 0 false alerts | 2 real detections | ✅ |
| Final calibrated | 632 | 0 false alerts | 2 real detections | ✅ Perfect |

#### Files created
| File | Purpose |
|---|---|
| `main.py` | Entry point — virtual/hardware mode switch, camera auto-detect, main loop |
| `core/detector.py` | EAR + MAR calculation, CNN inference, fusion logic, frame counter |
| `core/alert_manager.py` | Synthesized beep, OpenCV overlay, DB alert logging |
| `core/session_manager.py` | Session lifecycle — start, frame throttle, end |
| `database/db_queries.py` | All SQLite CRUD operations |
| `src/db_setup.py` | Updated schema with Day 3 column names |

#### Issues faced & resolved
| Issue | Fix |
|---|---|
| protobuf version conflict (mediapipe ImportError) | `pip install protobuf==3.20.3` |
| DB tables missing on first run | Run `python src/db_setup.py` first |
| DB column name mismatch (Day 1 vs Day 3) | Deleted old DB, updated db_setup.py schema to match db_queries.py |
| `model.predict()` crash in tight loop | Replaced with `model(img, training=False).numpy()` — direct call, no data pipeline |
| Pygame beep error "Array must be 2D for stereo" | Changed mixer to stereo, used `np.column_stack([mono, mono])` for stereo array |
| False alerts with eyes open (EAR ~0.30) | Lowered EAR_THRESHOLD to 0.20, raised CNN_THRESHOLD to 0.85 |
| Alert #5/#6 "NONE" type still firing | Increased frame decay from -1 to -3, removed CNN-alone eye closure rule |

---

### Day 4 — Hardware Mode (planned)
**Status:** 📋 Planned

- Raspberry Pi camera integration (picamera2)
- GPIO buzzer + LED alerts
- Same core/ logic as virtual mode — zero changes to detector.py
- Only main.py switches I/O layer

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
| model() over model.predict() | Direct call avoids Keras data pipeline — safe in real-time loops |
| EAR as ground truth for eyes | CNN can misfire on lighting/crop; EAR is geometry-based and reliable |
| Synthesized beep over wav file | No file dependency — tone generated in memory with numpy + pygame |
| CNN-alone eye rule removed | Fusion requires EAR + CNN agreement — eliminates false positives |

---

## Hardware Mode Notes (Day 4 reference)
When switching to hardware mode:
- Replace `cv2.VideoCapture(0)` with `picamera2` stream
- Replace `pygame.mixer` beep with `RPi.GPIO` buzzer on pin X
- Add LED indicator on GPIO pin Y
- SQLite DB path changes to `/home/pi/drowsiness.db`
- All `core/` files remain identical — zero changes needed

---

*Log maintained by: Austin Trinidad*
*Last updated: Day 3 — 2026-03-19*