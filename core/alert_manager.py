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
    Manages drowsiness alerts with graduated audio responses.
    Levels: 
      - low: Mild fatigue (two mid-pitch beeps)
      - high: High fatigue (triple fast beep)
      - critical: Emergency (siren effect)
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
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            return True
        except Exception as e:
            print(f"[ALERT] pygame not available: {e} — sound alerts disabled")
            return False

    # ─────────────────────────────────────────────
    # SOUND GENERATION
    # ─────────────────────────────────────────────

    def _generate_beep(self, frequency: int = 880, duration_ms: int = 800):
        """Synthesize a sine wave tone in memory."""
        if not self._pygame_ok:
            return
        try:
            sample_rate = 44100
            n_samples   = int(sample_rate * duration_ms / 1000)
            t           = np.linspace(0, duration_ms / 1000, n_samples, False)

            wave     = np.sin(2 * np.pi * frequency * t)
            envelope = np.ones(n_samples)
            attack   = int(sample_rate * 0.01)
            decay    = int(sample_rate * 0.1)
            envelope[:attack]  = np.linspace(0, 1, attack)
            envelope[-decay:]  = np.linspace(1, 0.3, decay)

            mono = (wave * envelope * 32767).astype(np.int16)
            stereo = np.column_stack([mono, mono])

            sound = self._pygame.sndarray.make_sound(stereo)
            sound.play()
            self._pygame.time.wait(duration_ms)
        except Exception as e:
            print(f"[ALERT] Beep error: {e}")

    # ─────────────────────────────────────────────
    # GRADUATED AUDIO LOGIC
    # ─────────────────────────────────────────────

    def _play_alert_sound(self, priority: str):
        """Play different beep patterns based on danger level."""
        if priority == "low":
            # Level 1: Mild Fatigue -> Two short, mid-pitch beeps
            self._generate_beep(frequency=600, duration_ms=200)
            time.sleep(0.1)
            self._generate_beep(frequency=600, duration_ms=200)
            
        elif priority == "high":
            # Level 2: High Fatigue -> Loud, fast triple beep
            for _ in range(3):
                self._generate_beep(frequency=900, duration_ms=250)
                time.sleep(0.1)
                
        elif priority == "critical":
            # Level 3: Critical Fatigue -> Continuous high-pitch siren effect
            for _ in range(5):
                self._generate_beep(frequency=1200, duration_ms=400)
                time.sleep(0.05)
        else:
            # Fallback (e.g., general yawning)
            self._generate_beep(frequency=880, duration_ms=300)

    def _play_sound_async(self, priority: str):
        """Trigger sound in background — doesn't block video loop."""
        if self._sound_thread is not None and self._sound_thread.is_alive():
            return 
        self._sound_thread = threading.Thread(
            target=self._play_alert_sound, args=(priority,), daemon=True
        )
        self._sound_thread.start()

    # ─────────────────────────────────────────────
    # TRIGGER
    # ─────────────────────────────────────────────

    def trigger(self, ear: float, mar: float,
                cnn_class: str, cnn_confidence: float, alert_type: str):
        """
        Fire an alert if cooldown has passed.
        Passes priority level to the async sound player.
        """
        now = time.time()
        if now - self.last_alert_time < self.ALERT_COOLDOWN:
            return

        self.last_alert_time = now
        self.alert_count    += 1

        # Passes priority level ('low', 'high', 'critical') to the sound system
        self._play_sound_async(priority=alert_type)

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
        """Draw live info panel + status banner onto the OpenCV frame."""
        h, w   = frame.shape[:2]
        is_alert = any(x in status for x in ["ALERT", "DROWSY", "CRITICAL", "HIGH", "LOW"])

        # ── Status banner (top) ──
        if status == "NO_FACE":
            banner_color = (80, 80, 80)
            banner_text  = "NO FACE DETECTED"
        elif is_alert:
            flash        = int(time.time() * 2) % 2 == 0
            banner_color = (0, 0, 220) if flash else (0, 0, 160)
            banner_text  = f"WARNING: {status.upper()}"
        else:
            banner_color = (30, 160, 30)
            banner_text  = "SYSTEM ACTIVE"

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

        # EAR & MAR Stats
        cv2.putText(frame, f"EAR: {ear:.3f}", (panel_x, panel_y + line_h * 0), font, fsmall, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (panel_x, panel_y + line_h * 1), font, fsmall, (255, 255, 255), 1)
        
        # CNN Confidence
        cv2.putText(frame, f"CNN: {cnn_class} {cnn_confidence:.0%}", (panel_x, panel_y + line_h * 2), font, fsmall, (0, 255, 255), 1)

        # Drowsy frame counter
        safe_threshold = max(frame_threshold, 1)
        progress       = min(frame_counter / safe_threshold, 1.0)
        cv2.putText(frame, f"Intensity: {frame_counter}/{frame_threshold}", (panel_x, panel_y + line_h * 3 + 16), font, fsmall, (200, 200, 200), 1)

        # Alert count + quit hint
        cv2.putText(frame, f"Alerts: {self.alert_count}", (w - 110, h - 15), font, fsmall, (200, 200, 200), 1)
        cv2.putText(frame, "Q = quit", (w - 90, 40), font, 0.45, (180, 180, 180), 1)

        return frame