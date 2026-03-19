"""
database/db_queries.py
======================
Day 3 — SQLite CRUD helpers
Driver Drowsiness Detection System

All database read/write operations go through here.
Used by session_manager.py and alert_manager.py.

Author: Austin Trinidad
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "drowsiness.db")


def get_connection():
    """Return a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safer for concurrent writes
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
# DRIVERS
# ─────────────────────────────────────────────

def get_or_create_driver(name: str = "Default Driver") -> int:
    """Return driver_id for the given name, creating if not exists."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM drivers WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return row["id"]
            cursor = conn.execute(
                "INSERT INTO drivers (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat())
            )
            return cursor.lastrowid
    except Exception as e:
        print(f"[DB] get_or_create_driver error: {e}")
        return 1  # fallback to driver id 1


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
# ALERTS
# ─────────────────────────────────────────────

def log_alert(session_id: int, alert_type: str, ear: float, mar: float,
              cnn_class: str, cnn_confidence: float):
    """
    Log a single drowsiness alert event.

    alert_type: 'closed_eye' | 'yawn' | 'combined'
    """
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


# ─────────────────────────────────────────────
# FRAME LOGS
# ─────────────────────────────────────────────

def log_frame(session_id: int, ear: float, mar: float,
              cnn_class: str, cnn_confidence: float, is_drowsy: bool):
    """
    Log per-frame detection data.
    Only called every N frames to avoid flooding the DB.
    """
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
# STATS (for future dashboard)
# ─────────────────────────────────────────────

def get_session_summary(session_id: int) -> dict:
    """Return alert count and duration for a session."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT start_time, end_time, total_alerts
                   FROM sessions WHERE id = ?""",
                (session_id,)
            ).fetchone()
            if row:
                return dict(row)
            return {}
    except Exception as e:
        print(f"[DB] get_session_summary error: {e}")
        return {}