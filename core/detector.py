"""
core/detector.py
================
Day 3 — Drowsiness Detector
Driver Drowsiness Detection System

Handles per-frame drowsiness detection:
  - MediaPipe face mesh → 468 landmarks
  - EAR (Eye Aspect Ratio) — detects eye closure
  - MAR (Mouth Aspect Ratio) — detects yawning
  - CNN model inference — classifies face crop
  - Fusion logic — combines EAR/MAR + CNN for final decision
  - Frame counter — N consecutive drowsy frames → ALERT

Author: Austin Trinidad
"""

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────

# Left eye landmarks (right eye from camera perspective)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks for MAR
MOUTH_MAR = [61, 291, 0, 17, 82, 312]

# Face bounding box landmarks (for CNN crop)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58,  132, 93,  234, 127,
             162, 21,  54,  103, 67,  109]


class DrowsinessDetector:
    """
    Per-frame drowsiness detector.

    Combines MediaPipe face landmarks + CNN model for robust detection.

    Thresholds (balanced mode ~2.5 sec alarm):
      EAR_THRESHOLD    : below this = eye closing
      MAR_THRESHOLD    : above this = yawning
      CNN_THRESHOLD    : confidence needed to count a CNN prediction
      FRAME_THRESHOLD  : consecutive drowsy frames before alarm
    """

    # ── Detection thresholds ──
    EAR_THRESHOLD   = 0.20   # below = closed/drooping eyes
    MAR_THRESHOLD   = 0.60   # above = yawning
    CNN_THRESHOLD   = 0.85   # min CNN confidence to act on prediction

    # Balanced mode: ~30 FPS × 0.08s ≈ 2.5 sec worth of frames
    FRAME_THRESHOLD = 30     # consecutive drowsy frames before alarm fires

    # CNN input size (must match train_model.py)
    CNN_IMG_SIZE = 96

    def __init__(self, model_path: str):
        # Load CNN model
        print(f"[DETECTOR] Loading CNN model: {model_path}")
        self.model = keras.models.load_model(model_path)

        # Warm up the model with a dummy prediction
        # This forces TF to compile the graph before the real-time loop starts
        # so the first frame doesn't stutter
        dummy = np.zeros((1, self.CNN_IMG_SIZE, self.CNN_IMG_SIZE, 3), dtype=np.float32)
        _ = self.model(dummy, training=False)
        print("[DETECTOR] CNN model loaded and warmed up.")

        # Class names — must match training order
        self.class_names = ["closed_eye", "open_eye", "yawn"]

        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Frame counter — consecutive frames where drowsiness detected
        self.drowsy_frame_count = 0

        # Last valid readings (shown on overlay when face briefly lost)
        self.last_ear       = 0.0
        self.last_mar       = 0.0
        self.last_cnn_class = "open_eye"
        self.last_cnn_conf  = 0.0

        print("[DETECTOR] MediaPipe face mesh ready.")

    # ─────────────────────────────────────────────
    # EAR / MAR CALCULATIONS
    # ─────────────────────────────────────────────

    @staticmethod
    def _euclidean(p1, p2) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _compute_ear(self, landmarks, indices: list, w: int, h: int) -> float:
        """
        Eye Aspect Ratio — measures how open the eye is.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        High EAR = open eye
        Low EAR  = closed eye (< EAR_THRESHOLD)
        """
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

        v1 = self._euclidean(pts[1], pts[5])
        v2 = self._euclidean(pts[2], pts[4])
        h1 = self._euclidean(pts[0], pts[3])

        if h1 < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h1)

    def _compute_mar(self, landmarks, indices: list, w: int, h: int) -> float:
        """
        Mouth Aspect Ratio — measures how open the mouth is.

        High MAR = open mouth / yawning (> MAR_THRESHOLD)
        Low MAR  = closed mouth
        """
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

        v1 = self._euclidean(pts[2], pts[5])
        v2 = self._euclidean(pts[3], pts[4])
        h1 = self._euclidean(pts[0], pts[1])

        if h1 < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h1)

    # ─────────────────────────────────────────────
    # CNN INFERENCE
    # ─────────────────────────────────────────────

    def _get_face_crop(self, frame: np.ndarray, landmarks, w: int, h: int):
        """
        Crop face region from frame using face oval landmarks.
        Returns resized 96x96 crop or None if crop fails.
        """
        try:
            xs = [int(landmarks[i].x * w) for i in FACE_OVAL]
            ys = [int(landmarks[i].y * h) for i in FACE_OVAL]

            x1 = max(0, min(xs) - int(w * 0.05))
            y1 = max(0, min(ys) - int(h * 0.05))
            x2 = min(w, max(xs) + int(w * 0.05))
            y2 = min(h, max(ys) + int(h * 0.05))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.CNN_IMG_SIZE, self.CNN_IMG_SIZE))
            return crop
        except Exception:
            return None

    def _cnn_predict(self, frame: np.ndarray, landmarks, w: int, h: int):
        """
        Run CNN on face crop using direct model call (faster than model.predict).
        Returns (class_name, confidence) or ('open_eye', 0.0) on failure.

        Uses model(img, training=False) instead of model.predict() —
        avoids Keras data pipeline overhead which causes crashes in tight loops.
        """
        crop = self._get_face_crop(frame, landmarks, w, h)
        if crop is None:
            return "open_eye", 0.0

        try:
            # Preprocess — same as training generator (rescale to [0,1])
            img = crop.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # (1, 96, 96, 3)

            # Direct model call — no data pipeline, safe in real-time loop
            probs      = self.model(img, training=False).numpy()[0]
            class_idx  = int(np.argmax(probs))
            confidence = float(probs[class_idx])

            return self.class_names[class_idx], confidence

        except Exception as e:
            print(f"[DETECTOR] CNN predict error: {e}")
            return "open_eye", 0.0

    # ─────────────────────────────────────────────
    # FUSION LOGIC
    # ─────────────────────────────────────────────

    def _is_drowsy_frame(self, ear, mar, cnn_class, cnn_conf):
      cnn_drowsy_eye  = (cnn_class == "closed_eye" and cnn_conf >= self.CNN_THRESHOLD)
      cnn_drowsy_yawn = (cnn_class == "yawn"        and cnn_conf >= self.CNN_THRESHOLD)
      ear_drowsy      = ear < self.EAR_THRESHOLD
      mar_drowsy      = mar > self.MAR_THRESHOLD

      # Eye closure — REQUIRE both CNN and EAR to agree (eliminates CNN false positives)
      if cnn_drowsy_eye and ear_drowsy:
        return True, "combined", "ALERT"

      # EAR alone is reliable enough — eyes clearly closing even if CNN unsure
      if ear_drowsy:
        return True, "closed_eye", "DROWSY"

      # Yawn
      if cnn_drowsy_yawn or (mar_drowsy and cnn_class != "open_eye"):
        return True, "yawn", "ALERT_YAWN"

      return False, "none", "OK"

    # ─────────────────────────────────────────────
    # MAIN PROCESS FRAME
    # ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single webcam frame end-to-end.

        Returns a result dict with: face_found, ear, mar, cnn_class,
        cnn_confidence, is_drowsy, alert_type, status,
        frame_counter, frame_threshold, should_alert
        """
        h, w = frame.shape[:2]

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face — gradually decay counter, return last known values
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 2)
            return {
                "face_found"     : False,
                "ear"            : self.last_ear,
                "mar"            : self.last_mar,
                "cnn_class"      : self.last_cnn_class,
                "cnn_confidence" : self.last_cnn_conf,
                "is_drowsy"      : False,
                "alert_type"     : "none",
                "status"         : "NO_FACE",
                "frame_counter"  : self.drowsy_frame_count,
                "frame_threshold": self.FRAME_THRESHOLD,
                "should_alert"   : False,
            }

        landmarks = results.multi_face_landmarks[0].landmark

        # EAR — average of both eyes
        ear_left  = self._compute_ear(landmarks, LEFT_EYE,  w, h)
        ear_right = self._compute_ear(landmarks, RIGHT_EYE, w, h)
        ear = (ear_left + ear_right) / 2.0

        # MAR
        mar = self._compute_mar(landmarks, MOUTH_MAR, w, h)

        # CNN
        cnn_class, cnn_conf = self._cnn_predict(frame, landmarks, w, h)

        # Cache last valid readings
        self.last_ear       = ear
        self.last_mar       = mar
        self.last_cnn_class = cnn_class
        self.last_cnn_conf  = cnn_conf

        # Fusion decision
        is_drowsy, alert_type, status = self._is_drowsy_frame(
            ear, mar, cnn_class, cnn_conf
        )

        # Frame counter
        if is_drowsy:
            self.drowsy_frame_count += 1
        else:
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 3)

        should_alert = self.drowsy_frame_count >= self.FRAME_THRESHOLD

        return {
            "face_found"     : True,
            "ear"            : ear,
            "mar"            : mar,
            "cnn_class"      : cnn_class,
            "cnn_confidence" : cnn_conf,
            "is_drowsy"      : is_drowsy,
            "alert_type"     : alert_type,
            "status"         : status if not should_alert else (
                                   "ALERT" if alert_type != "yawn" else "ALERT_YAWN"
                               ),
            "frame_counter"  : self.drowsy_frame_count,
            "frame_threshold": self.FRAME_THRESHOLD,
            "should_alert"   : should_alert,
        }

    def release(self):
        """Clean up MediaPipe resources."""
        self.face_mesh.close()
        print("[DETECTOR] Released.")