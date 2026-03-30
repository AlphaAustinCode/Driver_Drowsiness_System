import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "drowsiness.db")


def get_connection():
    """Return a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON") # Ensure FK constraints are active
    return conn


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def get_config(key: str, default=None):
    """Fetch a single config value by key."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
    except Exception as e:
        print(f"[DB] get_config error: {e}")
        return default


def set_config(key: str, value: str):
    """Update or insert a config value."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value))
            )
    except Exception as e:
        print(f"[DB] set_config error: {e}")


# ─────────────────────────────────────────────
# DRIVERS & PROFILES (Day 4)
# ─────────────────────────────────────────────

def get_or_create_driver(name: str = "Default Driver") -> int:
    """Return driver_id, creating driver and profile if not exists."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM drivers WHERE name = ?", (name,)
            ).fetchone()
            
            if row:
                return row["id"]
            
            # Create driver
            cursor = conn.execute(
                "INSERT INTO drivers (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat())
            )
            driver_id = cursor.lastrowid
            
            # Create associated profile
            conn.execute(
                "INSERT INTO driver_profiles (driver_id) VALUES (?)",
                (driver_id,)
            )
            return driver_id
    except Exception as e:
        print(f"[DB] get_or_create_driver error: {e}")
        return 1


def get_driver_profile(driver_id: int) -> Optional[Dict[str, Any]]:
    """Fetch sensitivity and trust data for a driver."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM driver_profiles WHERE driver_id = ?", (driver_id,)
            ).fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"[DB] get_driver_profile error: {e}")
        return None


def update_driver_profile(driver_id: int, sensitivity: float = None, trust: float = None):
    """Update profile metrics after a session or alert event."""
    try:
        with get_connection() as conn:
            if sensitivity is not None:
                conn.execute("UPDATE driver_profiles SET sensitivity_modifier = ?, updated_at = ? WHERE driver_id = ?",
                             (sensitivity, datetime.now().isoformat(), driver_id))
            if trust is not None:
                conn.execute("UPDATE driver_profiles SET trust_index = ?, updated_at = ? WHERE driver_id = ?",
                             (trust, datetime.now().isoformat(), driver_id))
    except Exception as e:
        print(f"[DB] update_driver_profile error: {e}")


# ─────────────────────────────────────────────
# FATIGUE EVENTS (Day 4)
# ─────────────────────────────────────────────

def log_fatigue_event(driver_id: int, level: int, score: float, circadian_label: str = None):
    """
    Log a high-level fatigue event.
    level: 1-5 (Scale of fatigue severity)
    score: Calculated fatigue probability
    """
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO fatigue_events 
                   (driver_id, fatigue_level, fatigue_score, circadian_label, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (driver_id, level, round(score, 4), circadian_label, datetime.now().isoformat())
            )
    except Exception as e:
        print(f"[DB] log_fatigue_event error: {e}")


# ─────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────

def start_session(driver_id: int) -> int:
    """Create a new session row and return its id."""
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (driver_id, start_time, status) VALUES (?, ?, ?)",
                (driver_id, datetime.now().isoformat(), "active")
            )
            session_id = cursor.lastrowid
            print(f"[DB] Session started — id: {session_id}")
            return session_id
    except Exception as e:
        print(f"[DB] start_session error: {e}")
        return -1


def end_session(session_id: int, total_alerts: int):
    """Mark a session as ended with summary stats."""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET end_time=?, status=?, total_alerts=? WHERE id=?",
                (datetime.now().isoformat(), "ended", total_alerts, session_id)
            )
            print(f"[DB] Session {session_id} ended — total alerts: {total_alerts}")
    except Exception as e:
        print(f"[DB] end_session error: {e}")


# ─────────────────────────────────────────────
# ALERTS & FRAME LOGS
# ─────────────────────────────────────────────

def log_alert(session_id: int, alert_type: str, ear: float, mar: float,
              cnn_class: str, cnn_confidence: float):
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO alerts
                   (session_id, timestamp, alert_type, ear_value, mar_value,
                    cnn_class, cnn_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, datetime.now().isoformat(), alert_type,
                 round(ear, 4), round(mar, 4), cnn_class, round(cnn_confidence, 4))
            )
    except Exception as e:
        print(f"[DB] log_alert error: {e}")


def log_frame(session_id: int, ear: float, mar: float,
              cnn_class: str, cnn_confidence: float, is_drowsy: bool):
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO frame_logs
                   (session_id, timestamp, ear_value, mar_value,
                    cnn_class, cnn_confidence, is_drowsy)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, datetime.now().isoformat(),
                 round(ear, 4), round(mar, 4),
                 cnn_class, round(cnn_confidence, 4), int(is_drowsy))
            )
    except Exception as e:
        print(f"[DB] log_frame error: {e}")


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def get_session_summary(session_id: int) -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT start_time, end_time, total_alerts
                   FROM sessions WHERE id = ?""",
                (session_id,)
            ).fetchone()
            return dict(row) if row else {}
    except Exception as e:
        print(f"[DB] get_session_summary error: {e}")
        return {}