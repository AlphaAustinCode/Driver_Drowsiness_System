import sys
import os
import argparse
import time
import cv2
import numpy as np
from datetime import datetime

# Add the project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.detector          import DrowsinessDetector
from core.alert_manager     import AlertManager
from core.session_manager    import SessionManager
from core.fatigue_classifier import FatigueClassifier 
from core.trust_engine      import TrustEngine
from core.baseline_calibrator import BaselineCalibrator
from core.cognitive_test    import CognitiveAssistant

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = os.path.join(BASE_DIR, "models", "drowsiness_model.keras")
WINDOW_NAME = "Driver Drowsiness Detection — Day 5"
DRIVER_NAME = "Default Driver"
DRIVER_ID   = 1 

# ─────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────

def find_camera(forced_index: int = None) -> cv2.VideoCapture:
    indices = [forced_index] if forced_index is not None else [1, 2]
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
        if cap: cap.release()
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
    print("Driver Drowsiness Detection — Virtual Mode (Day 5)")
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
    session_start = getattr(session, "start_time", datetime.now())

    detector    = DrowsinessDetector(model_path=MODEL_PATH)
    alert_mgr   = AlertManager(session_id=session.session_id)

    classifier  = FatigueClassifier(
        session_start = session_start,
        driver_id     = DRIVER_ID,
    )

    calibrator = BaselineCalibrator(driver_id=DRIVER_ID, required_frames=150)
    cog_assist = CognitiveAssistant()

    frame_idx    = 0
    fps_time     = time.time()
    fps_display  = 0.0
    total_frames = 0

    print("\n[LOOP] Detection running. Press Q to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[LOOP] Frame read failed.")
                break

            frame_idx    += 1
            total_frames += 1

            # ── Step 1: Raw detection (EAR / MAR / CNN) ──────────────────────
            result = detector.process_frame(frame)

            # ── CALIBRATION OVERRIDE ─────────────────────────────────────────
            if calibrator.is_calibrating and result.get("face_found"):
                just_finished = calibrator.update(result["ear"], result["mar"])
                
                if just_finished:
                    detector.EAR_THRESHOLD = calibrator.baseline_ear
                    detector.MAR_THRESHOLD = calibrator.baseline_mar
                    
                    # Update classifier constants globally
                    import core.fatigue_classifier as fc
                    fc.EAR_OPEN = calibrator.mean_ear
                    fc.MAR_CLOSED = calibrator.mean_mar
                    print(f"[CALIB] SUCCESS: New EAR Threshold set to {detector.EAR_THRESHOLD:.3f}")
                
                frame = calibrator.draw_overlay(frame)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue
            # ─────────────────────────────────────────────────────────────────

            # ── Step 2: Fatigue classification (Level 0–3) ───────────────────
            result = classifier.classify(result)

            # ── Step 2.5: Cognitive Assistance Logic ─────────────────────────
            # Trigger Stroop Test at Level 1 (Mild Fatigue)
            if result.get("fatigue_level") == 1:
                cog_assist.trigger()

            # Process the outcome of any ongoing or just-finished test
            test_outcome = cog_assist.pop_result()
            if test_outcome == "pass":
                print("[COG] Driver passed Stroop Test! Applying EMA reward.")
                classifier.apply_cognitive_reward()
            elif test_outcome in ["fail", "timeout"]:
                print(f"[COG] Driver {test_outcome}ed! Applying EMA penalty.")
                classifier.apply_cognitive_penalty()
                
                # SPECIAL CASE: If they timeout on a critical level, 
                # we force an immediate emergency SMS via the alert manager.
                if result.get("fatigue_level") == 3:
                     alert_mgr.trigger(
                        ear=result["ear"], mar=result["mar"],
                        cnn_class=result["cnn_class"], cnn_confidence=result["cnn_confidence"],
                        alert_type="critical", fatigue_score=1.0
                    )

            # ── DEBUG: Console Monitoring ────────────────────────────────────
            ema = getattr(classifier, '_ema_score', 0.0)
            is_drowsy_frame = result.get("is_drowsy", False) or result.get("fatigue_level", 0) > 0
            
            if (is_drowsy_frame or frame_idx % 20 == 0) and result.get("face_found"):
                print(
                    f"[CLASSIFIER] Raw={result.get('fatigue_score',0):.3f} "
                    f"EMA={ema:.3f} "
                    f"Level={result.get('fatigue_level',0)} ({result.get('level_label','?')}) | "
                    f"EAR={result.get('ear',0):.3f} | "
                    f"drowsy={result.get('is_drowsy',False)}"
                )

            # ── Step 3: Alert — Graduated Priority ───────────────────────────
            if result["should_alert"] and result["face_found"]:
                # UPDATED: Pass fatigue_score to trigger for SMS context
                alert_priority = result.get("level_label", "low").lower()
                alert_mgr.trigger(
                    ear            = result["ear"],
                    mar            = result["mar"],
                    cnn_class      = result["cnn_class"],
                    cnn_confidence = result["cnn_confidence"],
                    alert_type     = alert_priority,
                    fatigue_score  = result.get("fatigue_score", ema) # <── ADDED
                )

            # ── Step 4: Record to DB ─────────────────────────────────────────
            session.record_frame(
                ear            = result["ear"],
                mar            = result["mar"],
                cnn_class      = result["cnn_class"],
                cnn_confidence = result["cnn_confidence"],
                is_drowsy      = result["is_drowsy"],
            )

            # ── Step 5: Visual Overlay ───────────────────────────────────────
            frame = alert_mgr.draw_overlay(
                frame           = frame,
                status          = result.get("level_label", result["status"]),
                ear             = result["ear"],
                mar             = result["mar"],
                cnn_class       = result["cnn_class"],
                cnn_confidence  = result["cnn_confidence"],
                frame_counter   = result["frame_counter"],
                frame_threshold = result["frame_threshold"],
            )

            # Fatigue Badge UI
            level = result.get("fatigue_level", 0)
            level_label = result.get("level_label", "Alert")
            level_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 0, 255)}
            
            cv2.putText(frame, f"FATIGUE LEVEL {level}: {level_label}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_colors.get(level, (255,255,255)), 2)

            # Overlay Cognitive Test if active
            frame = cog_assist.draw_overlay(frame)

            # FPS Display
            if frame_idx % 30 == 0:
                fps_display = 30 / (time.time() - fps_time + 1e-6)
                fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (frame.shape[1]-100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[LOOP] Q pressed — stopping.")
                break

    except KeyboardInterrupt:
        print("\n[LOOP] Interrupted by user.")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    session.finish(total_alerts=alert_mgr.alert_count)
    session_end = datetime.now()

    # Calculate Trust Score
    print("\n[TRUST] Calculating session safety score...")
    try:
        trust_eng = TrustEngine()
        session_score = trust_eng.calculate_session_trust(
            driver_id=DRIVER_ID, 
            session_start=session_start, 
            session_end=session_end
        )
        trust_eng.update_driver_profile(DRIVER_ID, session_score)
        print(f"[TRUST] Session Score: {session_score:.2f} | Profile Updated.")
    except Exception as e:
        print(f"[TRUST] Post-session update failed: {e}")

    print(f"\n" + "="*60 + f"\n SESSION SUMMARY \n" + "="*60)
    print(f" Total frames   : {total_frames}")
    print(f" Total alerts   : {alert_mgr.alert_count}")
    print(f" Final level    : {classifier.get_current_level()}")
    print("="*60 + "\n")

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
        print("[HARDWARE] Hardware mode — starting Day 4 hardware sprint.")
    else:
        run_virtual_mode(cam_index=args.cam)