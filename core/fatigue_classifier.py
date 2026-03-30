import sqlite3
import os
import time
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "drowsiness.db"
)

# ── Circadian high-risk windows (24h clock) ──────────────────────────────────
# Based on human circadian biology — two natural sleepiness peaks
CIRCADIAN_HIGH_RISK_WINDOWS   = [(0, 6), (13, 15)]   # Midnight–6AM | 1PM–3PM
CIRCADIAN_MEDIUM_RISK_WINDOWS = [(6, 8), (22, 24)]   # Early morning | Late night

# ── Driving duration risk thresholds (minutes) ───────────────────────────────
DURATION_LOW_RISK    = 60    # Under 1 hour  → low risk
DURATION_MEDIUM_RISK = 120   # 1–2 hours     → medium risk
DURATION_HIGH_RISK   = 180   # Over 3 hours  → high risk

# ── Fatigue score level thresholds ───────────────────────────────────────────
LEVEL_1_THRESHOLD = 0.22    # Mild Fatigue
LEVEL_2_THRESHOLD = 0.45    # High Fatigue
LEVEL_3_THRESHOLD = 0.68    # Critical Fatigue

# ── Weighted contributions to fatigue score ───────────────────────────────────
WEIGHT_EAR       = 0.42    # Eye closure (primary signal)
WEIGHT_CNN       = 0.20    # CNN model prediction (secondary)
WEIGHT_MAR       = 0.15    # Yawning
WEIGHT_CIRCADIAN = 0.13    # Time-of-day circadian risk
WEIGHT_DURATION  = 0.10    # Driving duration risk

# ── EAR normalization bounds ─────────────────────────────────────────────────
EAR_OPEN   = 0.32          # Adjusted: typical open eye EAR
EAR_CLOSED = 0.15          # Typical closed eye EAR

# ── MAR normalization bounds ─────────────────────────────────────────────────
MAR_CLOSED = 0.30          # Closed mouth
MAR_YAWN   = 0.75          # Full yawn

# ── Circadian cache TTL ───────────────────────────────────────────────────────
CIRCADIAN_CACHE_TTL = 60.0  # Recalculate every 60 seconds


# ─────────────────────────────────────────────
# DB HELPER
# ─────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ─────────────────────────────────────────────
# 1. CIRCADIAN RISK ENGINE
# ─────────────────────────────────────────────

class CircadianRiskEngine:
    def __init__(self, session_start: Optional[datetime] = None):
        self.session_start   = session_start or datetime.now()
        self._cached_score   = 0.0
        self._cache_ts       = 0.0

    def get_risk(self) -> tuple:
        """Returns (risk_score: float, risk_label: str)"""
        now = time.time()
        if now - self._cache_ts < CIRCADIAN_CACHE_TTL and self._cache_ts > 0:
            return self._cached_score, self._to_label(self._cached_score)

        score = self._compute()
        self._cached_score = score
        self._cache_ts     = now
        return score, self._to_label(score)

    def get_sensitivity_multiplier(self) -> float:
        """Multiplier for detection sensitivity: 0.8 (low risk) → 1.4 (high risk)."""
        score, _ = self.get_risk()
        return round(0.8 + (score * 0.6), 3)

    def _compute(self) -> float:
        time_score     = self._time_of_day_score()
        duration_score = self._duration_score()
        combined = (time_score * 0.60) + (duration_score * 0.40)
        return round(min(max(combined, 0.0), 1.0), 4)

    def _time_of_day_score(self) -> float:
        hour = datetime.now().hour

        for start, end in CIRCADIAN_HIGH_RISK_WINDOWS:
            if start <= hour < end:
                mid      = (start + end) / 2.0
                halfspan = (end - start) / 2.0
                distance = abs(hour - mid) / halfspan
                return 1.0 - (distance * 0.25)

        for start, end in CIRCADIAN_MEDIUM_RISK_WINDOWS:
            if start <= hour < end:
                return 0.50

        return 0.15

    def _duration_score(self) -> float:
        elapsed = (datetime.now() - self.session_start).total_seconds() / 60.0

        if elapsed < DURATION_LOW_RISK:
            return (elapsed / DURATION_LOW_RISK) * 0.2
        elif elapsed < DURATION_MEDIUM_RISK:
            p = (elapsed - DURATION_LOW_RISK) / (DURATION_MEDIUM_RISK - DURATION_LOW_RISK)
            return 0.2 + (p * 0.3)
        elif elapsed < DURATION_HIGH_RISK:
            p = (elapsed - DURATION_MEDIUM_RISK) / (DURATION_HIGH_RISK - DURATION_MEDIUM_RISK)
            return 0.5 + (p * 0.3)
        else:
            extra = min((elapsed - DURATION_HIGH_RISK) / 60.0, 1.0)
            return 0.8 + (extra * 0.2)

    @staticmethod
    def _to_label(score: float) -> str:
        if score >= 0.65:
            return "high"
        elif score >= 0.35:
            return "medium"
        return "low"


# ─────────────────────────────────────────────
# 2. FATIGUE CLASSIFIER
# ─────────────────────────────────────────────

class FatigueClassifier:
    def __init__(
        self,
        session_start:  Optional[datetime] = None,
        driver_id:      int = 1,
    ):
        self.driver_id  = driver_id
        self.circadian  = CircadianRiskEngine(session_start=session_start)

        self._ema_score = 0.0
        self._spike_count = 0
        self._confirmed_level = 0
        self._last_level = 0
        self._history_modifier = self._load_history_modifier()

        print(f"[CLASSIFIER] Initialized. Driver ID: {driver_id}")
        print(f"[CLASSIFIER] History sensitivity modifier: {self._history_modifier:.2f}")

    def classify(self, detector_result: dict) -> dict:
        if not detector_result.get("face_found", False):
            self._decay_counters()
            return self._add_classifier_fields(detector_result, score=0.0, level=0)

        circ_score, circ_label = self.circadian.get_risk()
        sensitivity            = self.circadian.get_sensitivity_multiplier()

        ear_score  = self._score_ear(detector_result.get("ear", 0.3))
        mar_score  = self._score_mar(detector_result.get("mar", 0.2))
        cnn_score  = self._score_cnn(
            detector_result.get("cnn_class", "open_eye"),
            detector_result.get("cnn_confidence", 0.0)
        )
        dur_score  = self.circadian._duration_score()

        raw_score = (
            (ear_score  * WEIGHT_EAR)      +
            (cnn_score  * WEIGHT_CNN)      +
            (mar_score  * WEIGHT_MAR)      +
            (circ_score * WEIGHT_CIRCADIAN)+
            (dur_score  * WEIGHT_DURATION)
        )

        adjusted_score = min(raw_score * sensitivity * self._history_modifier, 1.0)
        adjusted_score = round(adjusted_score, 4)

        level = self._stabilize_level(adjusted_score)

        if level != self._last_level:
            self._log_level_transition(level, adjusted_score, circ_label)
            self._last_level = level

        return self._add_classifier_fields(
            detector_result,
            score      = adjusted_score,
            level      = level,
            circ_score = circ_score,
            circ_label = circ_label,
            sensitivity= sensitivity,
        )

    def get_current_level(self) -> int:
        return self._confirmed_level

    @staticmethod
    def _score_ear(ear: float) -> float:
        score = (EAR_OPEN - ear) / (EAR_OPEN - EAR_CLOSED)
        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def _score_mar(mar: float) -> float:
        score = (mar - MAR_CLOSED) / (MAR_YAWN - MAR_CLOSED)
        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def _score_cnn(cnn_class: str, confidence: float) -> float:
        if cnn_class == "open_eye":
            return 0.0
        elif cnn_class == "yawn":
            return round(confidence * 0.6, 4)
        elif cnn_class == "closed_eye":
            return round(confidence * 1.0, 4)
        return 0.0

    def _stabilize_level(self, score: float) -> int:
        ALPHA_UP   = 0.55
        ALPHA_DOWN = 0.04

        if score >= 0.45:
            self._spike_count += 1
        else:
            self._spike_count = max(0, self._spike_count - 1)

        if score > self._ema_score:
            self._ema_score = ALPHA_UP   * score + (1 - ALPHA_UP)   * self._ema_score
        else:
            self._ema_score = ALPHA_DOWN * score + (1 - ALPHA_DOWN) * self._ema_score

        self._ema_score = round(self._ema_score, 4)

        ema_level = 0
        if self._ema_score >= LEVEL_3_THRESHOLD:
            ema_level = 3
        elif self._ema_score >= LEVEL_2_THRESHOLD:
            ema_level = 2
        elif self._ema_score >= LEVEL_1_THRESHOLD:
            ema_level = 1

        spike_level = 1 if self._spike_count >= 2 else 0

        self._confirmed_level = max(ema_level, spike_level)
        return self._confirmed_level

    def _decay_counters(self):
        ALPHA_DOWN = 0.08
        self._ema_score = ALPHA_DOWN * 0.0 + (1 - ALPHA_DOWN) * self._ema_score
        self._ema_score = round(self._ema_score, 4)
        self._spike_count = 0
        if self._ema_score < LEVEL_1_THRESHOLD:
            self._confirmed_level = 0

    @staticmethod
    def _add_classifier_fields(
        result:      dict,
        score:       float,
        level:       int,
        circ_score:  float = 0.0,
        circ_label:  str   = "low",
        sensitivity: float = 1.0,
    ) -> dict:
        labels = {
            0: "Alert",
            1: "Mild Fatigue",
            2: "High Fatigue",
            3: "Critical Fatigue",
        }
        priorities = {
            0: "none",
            1: "low",
            2: "high",
            3: "critical",
        }

        result["fatigue_score"]           = score
        result["fatigue_level"]           = level
        result["level_label"]             = labels[level]
        result["circadian_score"]         = circ_score
        result["circadian_label"]         = circ_label
        result["sensitivity_multiplier"]  = sensitivity
        result["alert_priority"]          = priorities[level]

        result["should_alert"] = level >= 1 and result.get("face_found", False)

        if result.get("face_found", False):
            result["status"] = labels[level].upper().replace(" ", "_")

        return result

    def _load_history_modifier(self) -> float:
        try:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT sensitivity_modifier FROM driver_profiles WHERE driver_id = ?",
                (self.driver_id,)
            ).fetchone()
            conn.close()

            if row:
                modifier = float(row["sensitivity_modifier"])
                return min(max(modifier, 0.7), 1.5)
            return 1.0
        except Exception:
            return 1.0

    def _log_level_transition(self, new_level: int, score: float, circ_label: str):
        try:
            conn = _get_conn()
            conn.execute(
                """INSERT INTO fatigue_events
                   (driver_id, fatigue_level, fatigue_score, circadian_label, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    self.driver_id,
                    new_level,
                    score,
                    circ_label,
                    datetime.now().isoformat(),
                )
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# LEVEL DESCRIPTIONS (for AlertManager)
# ─────────────────────────────────────────────

LEVEL_DESCRIPTIONS = {
    0: {
        "label"       : "Alert",
        "color_bgr"   : (0, 200, 0),
        "action"      : "none",
    },
    1: {
        "label"       : "Mild Fatigue",
        "color_bgr"   : (0, 165, 255),
        "action"      : "voice_prompt",
    },
    2: {
        "label"       : "High Fatigue",
        "color_bgr"   : (0, 80, 255),
        "action"      : "loud_alarm",
    },
    3: {
        "label"       : "Critical Fatigue",
        "color_bgr"   : (0, 0, 255),
        "action"      : "emergency",
    },
}