"""
main.py
=======
Day 3 — Real-time Drowsiness Detection Entry Point
Driver Drowsiness Detection System

Runs the full real-time detection loop:
  - Auto-detects webcam
  - Processes each frame through DrowsinessDetector
  - Triggers alerts via AlertManager
  - Logs session + frames to SQLite via SessionManager

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

# Make sure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.detector        import DrowsinessDetector
from core.alert_manager   import AlertManager
from core.session_manager import SessionManager

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH   = os.path.join(BASE_DIR, "models", "drowsiness_model.keras")
WINDOW_NAME  = "Driver Drowsiness Detection"
DRIVER_NAME  = "Default Driver"

# Frame skip — run CNN every N frames to keep loop fast
# EAR/MAR still runs every frame; CNN is heavier
CNN_EVERY_N_FRAMES = 2


# ─────────────────────────────────────────────
# CAMERA AUTO-DETECT
# ─────────────────────────────────────────────

def find_camera(forced_index: int = None) -> cv2.VideoCapture:
    """
    Auto-detect available webcam.
    Tries indices 0, 1, 2 in order.
    Returns the first working VideoCapture.
    """
    indices = [forced_index] if forced_index is not None else [0, 1, 2]

    for idx in indices:
        print(f"[CAM] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAM] Camera {idx} found — {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                return cap
        cap.release()

    print("[CAM] No camera found. Check your webcam connection.")
    sys.exit(1)


def configure_camera(cap: cv2.VideoCapture):
    """Set camera resolution and FPS."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)


# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────

def run_virtual_mode(cam_index: int = None):
    """
    Main real-time detection loop for virtual (webcam) mode.

    Loop:
      read frame → detect → draw overlay → show → handle alerts → repeat
    """
    print("\n" + "="*60)
    print("Driver Drowsiness Detection — Virtual Mode")
    print("="*60)
    print("Press Q to quit.\n")

    # Validate model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run src/train_model.py first.")
        sys.exit(1)

    # Init all components
    cap             = find_camera(cam_index)
    configure_camera(cap)

    session         = SessionManager(driver_name=DRIVER_NAME)
    session.begin()

    detector        = DrowsinessDetector(model_path=MODEL_PATH)
    alert_mgr       = AlertManager(session_id=session.session_id)

    frame_idx       = 0
    fps_time        = time.time()
    fps_display     = 0.0
    total_frames    = 0

    # Cache last CNN result so we can skip CNN on some frames
    last_cnn_result = {"cnn_class": "open_eye", "cnn_confidence": 0.0}

    print("\n[LOOP] Detection running. Press Q in the window to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[LOOP] Frame read failed — camera disconnected?")
            break

        frame_idx   += 1
        total_frames += 1

        # ── Detection ──
        result = detector.process_frame(frame)

        # ── Alert trigger ──
        if result["should_alert"] and result["face_found"]:
            alert_mgr.trigger(
                ear=result["ear"],
                mar=result["mar"],
                cnn_class=result["cnn_class"],
                cnn_confidence=result["cnn_confidence"],
                alert_type=result["alert_type"],
            )

        # ── Session frame log ──
        session.record_frame(
            ear=result["ear"],
            mar=result["mar"],
            cnn_class=result["cnn_class"],
            cnn_confidence=result["cnn_confidence"],
            is_drowsy=result["is_drowsy"],
        )

        # ── Draw overlay ──
        frame = alert_mgr.draw_overlay(
            frame=frame,
            status=result["status"],
            ear=result["ear"],
            mar=result["mar"],
            cnn_class=result["cnn_class"],
            cnn_confidence=result["cnn_confidence"],
            frame_counter=result["frame_counter"],
            frame_threshold=result["frame_threshold"],
        )

        # ── FPS counter (top-right) ──
        if frame_idx % 30 == 0:
            now = time.time()
            fps_display = 30 / (now - fps_time + 1e-6)
            fps_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (frame.shape[1] - 100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 160, 160), 1, cv2.LINE_AA)

        # ── Show frame ──
        cv2.imshow(WINDOW_NAME, frame)

        # ── Quit on Q ──
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[LOOP] Q pressed — stopping.")
            break

        # ── Also quit if window is closed ──
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[LOOP] Window closed — stopping.")
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    session.finish(total_alerts=alert_mgr.alert_count)

    print(f"\n[DONE] Session complete.")
    print(f"[DONE] Total frames processed : {total_frames}")
    print(f"[DONE] Total alerts triggered  : {alert_mgr.alert_count}")
    print(f"[DONE] Session logged to       : drowsiness.db")


# ─────────────────────────────────────────────
# HARDWARE MODE STUB (Day 4)
# ─────────────────────────────────────────────

def run_hardware_mode():
    """
    Hardware mode — Raspberry Pi camera + GPIO buzzer.
    Implemented in Day 4. Same core/ logic, different I/O.
    """
    print("[HARDWARE] Hardware mode not yet implemented — coming Day 4.")
    print("[HARDWARE] Replace cv2.VideoCapture with picamera2 stream.")
    print("[HARDWARE] Replace pygame beep with RPi.GPIO buzzer.")
    sys.exit(0)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System"
    )
    parser.add_argument(
        "--mode", choices=["virtual", "hardware"], default="virtual",
        help="Run mode: virtual (webcam) or hardware (Pi). Default: virtual"
    )
    parser.add_argument(
        "--cam", type=int, default=None,
        help="Force a specific camera index (0, 1, 2...). Default: auto-detect"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "hardware":
        run_hardware_mode()
    else:
        run_virtual_mode(cam_index=args.cam)