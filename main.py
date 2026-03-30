"""
main.py
=======
Day 4 — Real-time Drowsiness Detection Entry Point (Updated)
Driver Drowsiness Detection System

Changes from Day 3:
  - FatigueClassifier inserted between detector and alert_mgr
  - result dict now carries fatigue_level (0–3) and level_label
  - Alert priority passed to AlertManager for graduated response
  - Overlay shows fatigue level instead of raw status

Usage:
    python main.py                  # virtual mode (webcam)
    python main.py --mode hardware  # hardware mode (Day 4)
    python main.py --cam 1          # force camera index 1

Author: Austin Trinidad
"""

import sys
import os
import argparse
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.detector           import DrowsinessDetector
from core.alert_manager      import AlertManager
from core.session_manager    import SessionManager
from core.fatigue_classifier import FatigueClassifier   # ← NEW Day 4

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = os.path.join(BASE_DIR, "models", "drowsiness_model.keras")
WINDOW_NAME = "Driver Drowsiness Detection"
DRIVER_NAME = "Default Driver"
DRIVER_ID   = 1   # Default driver ID in DB


# ─────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────

def find_camera(forced_index: int = None) -> cv2.VideoCapture:
    indices = [forced_index] if forced_index is not None else [0, 1, 2]
    for idx in indices:
        print(f"[CAM] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAM] Camera {idx} found — "
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                return cap
        cap.release()
    print("[CAM] No camera found.")
    sys.exit(1)


def configure_camera(cap: cv2.VideoCapture):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)


# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────

def run_virtual_mode(cam_index: int = None):
    print("\n" + "="*60)
    print("Driver Drowsiness Detection — Virtual Mode (Day 4)")
    print("="*60)
    print("Press Q to quit.\n")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    # ── Init components ──────────────────────────────────────────────────────
    cap         = find_camera(cam_index)
    configure_camera(cap)

    session     = SessionManager(driver_name=DRIVER_NAME)
    session.begin()
    session_start = session.start_time if hasattr(session, "start_time") else None

    detector    = DrowsinessDetector(model_path=MODEL_PATH)
    alert_mgr   = AlertManager(session_id=session.session_id)

    # ── FatigueClassifier — NEW Day 4 ────────────────────────────────────────
    classifier  = FatigueClassifier(
        session_start = session_start,
        driver_id     = DRIVER_ID,
    )

    frame_idx    = 0
    fps_time     = time.time()
    fps_display  = 0.0
    total_frames = 0

    print("\n[LOOP] Detection running. Press Q to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[LOOP] Frame read failed.")
            break

        frame_idx    += 1
        total_frames += 1

        # ── Step 1: Raw detection (EAR / MAR / CNN) ──────────────────────────
        result = detector.process_frame(frame)

        # ── Step 2: Fatigue classification (Level 0–3) ← NEW ─────────────────
        result = classifier.classify(result)

        # ── DEBUG: print on every drowsy frame + every 20 frames normally ──────
        ema = getattr(classifier, '_ema_score', 0.0)
        is_drowsy_frame = result.get("is_drowsy", False) or result.get("fatigue_level", 0) > 0
        if (is_drowsy_frame or frame_idx % 20 == 0) and result.get("face_found"):
            print(
                f"[CLASSIFIER] Raw={result.get('fatigue_score',0):.3f} "
                f"EMA={ema:.3f} "
                f"Level={result.get('fatigue_level',0)} ({result.get('level_label','?')}) | "
                f"EAR={result.get('ear',0):.3f} "
                f"CNN={result.get('cnn_class','?')} {result.get('cnn_confidence',0)*100:.1f}% | "
                f"drowsy={result.get('is_drowsy',False)}"
            )

        # ── Step 3: Alert — now uses fatigue level priority ───────────────────
        if result["should_alert"] and result["face_found"]:
            alert_mgr.trigger(
                ear            = result["ear"],
                mar            = result["mar"],
                cnn_class      = result["cnn_class"],
                cnn_confidence = result["cnn_confidence"],
                alert_type     = result.get("alert_priority", result["alert_type"]),
            )

        # ── Step 4: Log frame ─────────────────────────────────────────────────
        session.record_frame(
            ear            = result["ear"],
            mar            = result["mar"],
            cnn_class      = result["cnn_class"],
            cnn_confidence = result["cnn_confidence"],
            is_drowsy      = result["is_drowsy"],
        )

        # ── Step 5: Draw overlay with fatigue level ───────────────────────────
        frame = alert_mgr.draw_overlay(
            frame           = frame,
            status          = result["status"],
            ear             = result["ear"],
            mar             = result["mar"],
            cnn_class       = result["cnn_class"],
            cnn_confidence  = result["cnn_confidence"],
            frame_counter   = result["frame_counter"],
            frame_threshold = result["frame_threshold"],
        )

        # ── Draw fatigue level badge (top-left) ───────────────────────────────
        level       = result.get("fatigue_level", 0)
        level_label = result.get("level_label", "Alert")
        level_colors = {
            0: (0, 200, 0),      # Green
            1: (0, 165, 255),    # Orange
            2: (0, 80, 255),     # Red-orange
            3: (0, 0, 255),      # Red
        }
        badge_color = level_colors.get(level, (0, 200, 0))
        cv2.putText(
            frame,
            f"Level {level}: {level_label}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            badge_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Score: {result.get('fatigue_score', 0.0):.2f}",
            (10, frame.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA
        )

        # ── FPS counter ───────────────────────────────────────────────────────
        if frame_idx % 30 == 0:
            now = time.time()
            fps_display = 30 / (now - fps_time + 1e-6)
            fps_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (frame.shape[1] - 100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[LOOP] Q pressed — stopping.")
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[LOOP] Window closed — stopping.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    session.finish(total_alerts=alert_mgr.alert_count)

    print(f"\n[DONE] Session complete.")
    print(f"[DONE] Total frames   : {total_frames}")
    print(f"[DONE] Total alerts   : {alert_mgr.alert_count}")
    print(f"[DONE] Final level    : {classifier.get_current_level()}")


# ─────────────────────────────────────────────
# HARDWARE MODE STUB
# ─────────────────────────────────────────────

def run_hardware_mode():
    print("[HARDWARE] Hardware mode — coming Day 4 hardware sprint.")
    sys.exit(0)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--mode", choices=["virtual", "hardware"], default="virtual")
    parser.add_argument("--cam",  type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "hardware":
        run_hardware_mode()
    else:
        run_virtual_mode(cam_index=args.cam)