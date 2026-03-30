import numpy as np
import sqlite3
import cv2

class BaselineCalibrator:
    def __init__(self, driver_id=1, required_frames=150, db_path="drowsiness.db"):
        self.driver_id = driver_id
        # 150 frames @ 30fps = 5 seconds of calibration (you can increase this later)
        self.required_frames = required_frames 
        self.db_path = db_path
        
        self.ear_history = []
        self.mar_history = []
        self.is_calibrating = True
        
        # Final calculated metrics
        self.mean_ear = None
        self.mean_mar = None
        self.baseline_ear = None
        self.baseline_mar = None

        self._ensure_db_columns()

    def _ensure_db_columns(self):
        """Safely add baseline columns to driver_profiles if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute("ALTER TABLE driver_profiles ADD COLUMN baseline_ear REAL")
            cur.execute("ALTER TABLE driver_profiles ADD COLUMN baseline_mar REAL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Columns already exist
        conn.close()

    def update(self, ear: float, mar: float) -> bool:
        """
        Feeds new readings into the history. 
        Returns True ONLY on the exact frame calibration finishes.
        """
        if not self.is_calibrating:
            return False

        self.ear_history.append(ear)
        self.mar_history.append(mar)

        if len(self.ear_history) >= self.required_frames:
            self._compute_baselines()
            return True
            
        return False

    def _compute_baselines(self):
        """Calculates Mean and Standard Deviation to set thresholds."""
        ear_arr = np.array(self.ear_history)
        mar_arr = np.array(self.mar_history)

        self.mean_ear = np.mean(ear_arr)
        self.mean_mar = np.mean(mar_arr)

        # ── The Math ──
        # EAR drops when sleepy -> Mean MINUS 2 Standard Deviations
        self.baseline_ear = round(self.mean_ear - (2 * np.std(ear_arr)), 3)
        
        # MAR rises when yawning -> Mean PLUS 3 Standard Deviations
        self.baseline_mar = round(self.mean_mar + (3 * np.std(mar_arr)), 3)

        # Clamp to safe extremes so a bad calibration doesn't break the system
        self.baseline_ear = max(0.15, min(self.baseline_ear, 0.28))
        self.baseline_mar = max(0.40, min(self.baseline_mar, 0.80))

        self._save_to_db()
        self.is_calibrating = False

    def _save_to_db(self):
        """Updates the driver profile with personalized thresholds."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """UPDATE driver_profiles 
               SET baseline_ear = ?, baseline_mar = ? 
               WHERE driver_id = ?""",
            (self.baseline_ear, self.baseline_mar, self.driver_id)
        )
        conn.commit()
        conn.close()
        print(f"\n[CALIBRATION COMPLETE]")
        print(f" -> Normal EAR: {self.mean_ear:.3f} | Threshold: {self.baseline_ear}")
        print(f" -> Normal MAR: {self.mean_mar:.3f} | Threshold: {self.baseline_mar}\n")

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws a 'Calibrating' progress bar on the screen."""
        h, w = frame.shape[:2]
        progress = len(self.ear_history) / self.required_frames
        
        # Top Banner
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        cv2.putText(frame, "CALIBRATING DRIVER BASELINE...", (w//2 - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Please look straight ahead naturally", (w//2 - 160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Progress bar filling up
        cv2.rectangle(frame, (0, 70), (w, 80), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, 70), (int(w * progress), 80), (0, 200, 0), -1)
        
        return frame