import sys
import os
import time
import threading
import numpy as np
import cv2

# Ensure core and database modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_queries import log_alert
from core.api_services import APIManager

class AlertManager:
    """
    Manages drowsiness alerts with graduated audio responses and API integrations.
    Levels: 
      - low: Mild fatigue (two mid-pitch beeps)
      - high: High fatigue (triple fast beep + Google Maps API rest stop search)
      - critical: Emergency (siren effect + Twilio SMS Alert)
    """

    # Cooldown between consecutive alerts (seconds) to prevent spamming
    ALERT_COOLDOWN = 4.0

    def __init__(self, session_id: int):
        self.session_id      = session_id
        self.last_alert_time = 0.0
        self.alert_count     = 0
        self._sound_thread   = None
        self._pygame_ok      = self._init_pygame()
        
        # API Integration State
        self.api = APIManager()
        self.suggested_destination = None

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
    # SOUND GENERATION (Digital Synthesis)
    # ─────────────────────────────────────────────

    def _generate_beep(self, frequency: int = 880, duration_ms: int = 800):
        """Synthesize a sine wave tone in memory to avoid external file dependencies."""
        if not self._pygame_ok:
            return
        try:
            sample_rate = 44100
            n_samples   = int(sample_rate * duration_ms / 1000)
            t           = np.linspace(0, duration_ms / 1000, n_samples, False)

            wave     = np.sin(2 * np.pi * frequency * t)
            envelope = np.ones(n_samples)
            
            # Simple ADSR envelope to prevent clicking sounds
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
            self._generate_beep(frequency=880, duration_ms=300)

    def _play_sound_async(self, priority: str):
        """Trigger sound in background — doesn't block the CV processing loop."""
        if self._sound_thread is not None and self._sound_thread.is_alive():
            return 
        self._sound_thread = threading.Thread(
            target=self._play_alert_sound, args=(priority,), daemon=True
        )
        self._sound_thread.start()

    # ─────────────────────────────────────────────
    # TRIGGER & API BRIDGE
    # ─────────────────────────────────────────────

    def trigger(self, ear: float, mar: float,
                cnn_class: str, cnn_confidence: float, alert_type: str, fatigue_score: float = 0.0):
        
        now = time.time()
        if now - self.last_alert_time < self.ALERT_COOLDOWN:
            return

        self.last_alert_time = now
        self.alert_count    += 1

        self._play_sound_async(priority=str(alert_type).lower())

        # ── API TRIGGER (BASED ON RAW SCORE) ──
        # We stop relying on strings. If the score is high enough, we fire.
        print(f"[DEBUG] Alert Manager triggered with score: {fatigue_score}")
        
        if fatigue_score >= 0.68: # Level 3 Threshold
            print("\n[API] 🚨 LEVEL 3 CONFIRMED! Sending Twilio SMS...")
            self.api.send_emergency_sms_async(driver_id=1, score=f"{fatigue_score:.2f}")
            
        elif fatigue_score >= 0.45: # Level 2 Threshold
            print(f"\n[API] 📍 LEVEL 2 CONFIRMED! Fetching Rest Stop...")
            def set_destination(name):
                self.suggested_destination = name
            self.api.fetch_rest_stop_async(callback=set_destination)
        # ──────────────────────────────────────

        # Log to Database
        if self.session_id and self.session_id > 0:
            log_alert(
                session_id=self.session_id,
                alert_type=str(alert_type),
                ear=ear,
                mar=mar,
                cnn_class=cnn_class,
                cnn_confidence=cnn_confidence
            )

        print(f"[ALERT #{self.alert_count}] {str(alert_type).upper()} | "
              f"EAR={ear:.3f} MAR={mar:.3f} | "
              f"CNN={cnn_class} {cnn_confidence:.1%}")

    # ─────────────────────────────────────────────
    # UI OVERLAY
    # ─────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray,
                     status: str,
                     ear: float,
                     mar: float,
                     cnn_class: str,
                     cnn_confidence: float,
                     frame_counter: int,
                     frame_threshold: int) -> np.ndarray:
        """Draw live info panel + status banner + API suggestions onto the frame."""
        h, w   = frame.shape[:2]
        is_alert = any(x in status for x in ["ALERT", "DROWSY", "CRITICAL", "HIGH", "LOW"])

        # ── Banner (Top) ──
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
        cv2.putText(frame, banner_text, (w // 2 - len(banner_text) * 8, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Navigation Suggestion (API Result) ──
        if self.suggested_destination:
            # Displayed above the bottom panel
            cv2.rectangle(frame, (10, h - 195), (w - 10, h - 165), (0, 0, 0), -1)
            cv2.putText(frame, f"SUGGESTED STOP: {self.suggested_destination}", 
                        (15, h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ── Info Panel (Bottom-Left) ──
        panel_x, panel_y = 10, h - 160
        cv2.rectangle(frame, (panel_x - 5, panel_y - 10), (300, h - 5), (20, 20, 20), -1)
        cv2.rectangle(frame, (panel_x - 5, panel_y - 10), (300, h - 5), (80, 80, 80), 1)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fsmall = 0.52
        line_h = 26

        cv2.putText(frame, f"EAR: {ear:.3f}", (panel_x, panel_y + line_h * 0), font, fsmall, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (panel_x, panel_y + line_h * 1), font, fsmall, (255, 255, 255), 1)
        cv2.putText(frame, f"CNN: {cnn_class} {cnn_confidence:.0%}", (panel_x, panel_y + line_h * 2), font, fsmall, (0, 255, 255), 1)
        
        # Intensity Bar logic
        cv2.putText(frame, f"Intensity: {frame_counter}/{frame_threshold}", 
                    (panel_x, panel_y + line_h * 3 + 16), font, fsmall, (200, 200, 200), 1)

        # ── Global Metadata ──
        cv2.putText(frame, f"Alerts: {self.alert_count}", (w - 110, h - 15), font, fsmall, (200, 200, 200), 1)
        cv2.putText(frame, "Q = quit", (w - 90, 40), font, 0.45, (180, 180, 180), 1)

        return frame