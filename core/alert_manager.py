"""
core/alert_manager.py
=====================
Day 3 — Alert Manager
Driver Drowsiness Detection System

Handles all alert outputs when drowsiness is detected:
  1. Beep sound (generated in code — no wav file needed)
  2. Visual overlay on OpenCV frame (red flash + warning text)
  3. SQLite alert logging via db_queries

Author: Austin Trinidad
"""

import sys
import os
import time
import threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_queries import log_alert


class AlertManager:
    """
    Manages drowsiness alerts — sound, visual, and DB logging.

    Usage:
        am = AlertManager(session_id=1)
        am.trigger(ear, mar, cnn_class, cnn_confidence, alert_type)
        am.draw_overlay(frame, status, ear, mar, cnn_class, cnn_confidence, frame_counter, threshold)
    """

    # Cooldown between consecutive alerts (seconds)
    ALERT_COOLDOWN = 4.0

    def __init__(self, session_id: int):
        self.session_id      = session_id
        self.last_alert_time = 0.0
        self.alert_count     = 0
        self._sound_thread   = None
        self._pygame_ok      = self._init_pygame()

    def _init_pygame(self) -> bool:
        """Initialize pygame mixer for beep generation."""
        try:
            import pygame
            # channels=2 for stereo — numpy array must be 2D (n_samples, 2)
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            return True
        except Exception as e:
            print(f"[ALERT] pygame not available: {e} — sound alerts disabled")
            return False

    # ─────────────────────────────────────────────
    # SOUND
    # ─────────────────────────────────────────────

    def _generate_beep(self, frequency: int = 880, duration_ms: int = 800):
        """
        Generate and play a beep tone using numpy + pygame.
        No wav file needed — tone synthesized in memory.

        Stereo fix: pygame stereo mixer requires shape (n_samples, 2).
        """
        if not self._pygame_ok:
            return
        try:
            sample_rate = 44100
            n_samples   = int(sample_rate * duration_ms / 1000)
            t           = np.linspace(0, duration_ms / 1000, n_samples, False)

            # Sine wave with attack/decay envelope
            wave     = np.sin(2 * np.pi * frequency * t)
            envelope = np.ones(n_samples)
            attack   = int(sample_rate * 0.01)
            decay    = int(sample_rate * 0.1)
            envelope[:attack]  = np.linspace(0, 1, attack)
            envelope[-decay:]  = np.linspace(1, 0.3, decay)

            # Mono samples scaled to int16
            mono = (wave * envelope * 32767).astype(np.int16)

            # Stereo: duplicate mono into shape (n_samples, 2)
            stereo = np.column_stack([mono, mono])

            sound = self._pygame.sndarray.make_sound(stereo)
            sound.play()
            self._pygame.time.wait(duration_ms)
        except Exception as e:
            print(f"[ALERT] Beep error: {e}")

    def _play_alert_sound(self):
        """Play double-beep pattern."""
        self._generate_beep(frequency=880,  duration_ms=300)
        time.sleep(0.15)
        self._generate_beep(frequency=1050, duration_ms=400)

    def _play_sound_async(self):
        """Trigger sound in background — doesn't block video loop."""
        if self._sound_thread is not None and self._sound_thread.is_alive():
            return  # don't stack sounds
        self._sound_thread = threading.Thread(
            target=self._play_alert_sound, daemon=True
        )
        self._sound_thread.start()

    # ─────────────────────────────────────────────
    # TRIGGER
    # ─────────────────────────────────────────────

    def trigger(self, ear: float, mar: float,
                cnn_class: str, cnn_confidence: float, alert_type: str):
        """
        Fire an alert if cooldown has passed.
        Plays sound + logs to DB.
        """
        now = time.time()
        if now - self.last_alert_time < self.ALERT_COOLDOWN:
            return

        self.last_alert_time = now
        self.alert_count    += 1

        self._play_sound_async()

        if self.session_id and self.session_id > 0:
            log_alert(
                session_id=self.session_id,
                alert_type=alert_type,
                ear=ear,
                mar=mar,
                cnn_class=cnn_class,
                cnn_confidence=cnn_confidence
            )

        print(f"[ALERT #{self.alert_count}] {alert_type.upper()} | "
              f"EAR={ear:.3f} MAR={mar:.3f} | "
              f"CNN={cnn_class} {cnn_confidence:.1%}")

    # ─────────────────────────────────────────────
    # OVERLAY
    # ─────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray,
                     status: str,
                     ear: float,
                     mar: float,
                     cnn_class: str,
                     cnn_confidence: float,
                     frame_counter: int,
                     frame_threshold: int) -> np.ndarray:
        """
        Draw live info panel + status banner onto the OpenCV frame.
        Returns the annotated frame.
        """
        h, w   = frame.shape[:2]
        is_alert = status.startswith("ALERT") or status == "DROWSY"

        # ── Status banner (top) ──
        if status == "NO_FACE":
            banner_color = (80, 80, 80)
            banner_text  = "NO FACE DETECTED"
        elif is_alert:
            flash        = int(time.time() * 2) % 2 == 0
            banner_color = (0, 0, 220) if flash else (0, 0, 160)
            label_map    = {
                "ALERT":      "DROWSINESS DETECTED",
                "ALERT_YAWN": "YAWNING DETECTED",
                "DROWSY":     "EYES CLOSING",
            }
            banner_text = label_map.get(status, "ALERT")
        else:
            banner_color = (30, 160, 30)
            banner_text  = "ALERT"

        cv2.rectangle(frame, (0, 0), (w, 50), banner_color, -1)
        cv2.putText(frame, banner_text,
                    (w // 2 - len(banner_text) * 8, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ── Info panel (bottom-left) ──
        panel_x, panel_y = 10, h - 160
        cv2.rectangle(frame, (panel_x - 5, panel_y - 10),
                      (300, h - 5), (20, 20, 20), -1)
        cv2.rectangle(frame, (panel_x - 5, panel_y - 10),
                      (300, h - 5), (80, 80, 80), 1)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fsmall = 0.52
        line_h = 26

        # EAR
        ear_color = (50, 50, 220) if ear < 0.25 else (50, 220, 50)
        cv2.putText(frame, f"EAR: {ear:.3f}",
                    (panel_x, panel_y + line_h * 0),
                    font, fsmall, ear_color, 1, cv2.LINE_AA)

        # MAR
        mar_color = (50, 180, 220) if mar > 0.6 else (50, 220, 50)
        cv2.putText(frame, f"MAR: {mar:.3f}",
                    (panel_x, panel_y + line_h * 1),
                    font, fsmall, mar_color, 1, cv2.LINE_AA)

        # CNN class + confidence
        cnn_color_map = {
            "closed_eye": (50,  50,  220),
            "yawn":       (50,  180, 220),
            "open_eye":   (50,  220, 50),
        }
        cnn_color = cnn_color_map.get(cnn_class, (200, 200, 200))
        cv2.putText(frame, f"CNN: {cnn_class} {cnn_confidence:.0%}",
                    (panel_x, panel_y + line_h * 2),
                    font, fsmall, cnn_color, 1, cv2.LINE_AA)

        # Confidence bar
        bar_x         = panel_x
        bar_y         = panel_y + line_h * 3 - 8
        bar_w_total   = 200
        bar_w_filled  = int(bar_w_total * cnn_confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w_total, bar_y + 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w_filled, bar_y + 8), cnn_color, -1)

        # Drowsy frame counter
        safe_threshold = max(frame_threshold, 1)
        progress       = min(frame_counter / safe_threshold, 1.0)
        bar_w2         = int(bar_w_total * progress)
        prog_color     = (0, 0, 220) if progress > 0.7 else \
                         (0, 165, 255) if progress > 0.4 else (0, 220, 0)
        cv2.putText(frame, f"Drowsy frames: {frame_counter}/{frame_threshold}",
                    (panel_x, panel_y + line_h * 3 + 16),
                    font, fsmall, (200, 200, 200), 1, cv2.LINE_AA)
        bar_y2 = panel_y + line_h * 4 - 4
        cv2.rectangle(frame, (bar_x, bar_y2 + 10),
                      (bar_x + bar_w_total, bar_y2 + 18), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y2 + 10),
                      (bar_x + bar_w2,      bar_y2 + 18), prog_color, -1)

        # Alert count + quit hint
        cv2.putText(frame, f"Alerts: {self.alert_count}",
                    (w - 110, h - 15),
                    font, fsmall, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q = quit",
                    (w - 90, 40),
                    font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return frame