"""
db_setup.py — Day 1: Database Setup
Driver Drowsiness Detection System
SQLite | Schema + Indexes + Seed Data + Migration Support
"""

import sqlite3
import os
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DB_PATH = "drowsiness.db"
SCHEMA_VERSION = 1  # Bump this when you add migrations


# ──────────────────────────────────────────────
# CONNECTION HELPER
# ──────────────────────────────────────────────
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # access columns by name
    conn.execute("PRAGMA foreign_keys = ON") # enforce FK constraints
    conn.execute("PRAGMA journal_mode = WAL") # better concurrent reads
    return conn


# ──────────────────────────────────────────────
# SCHEMA
# ──────────────────────────────────────────────
SCHEMA_SQL = """
-- ── drivers ──────────────────────────────────
-- One row per registered driver / user profile
CREATE TABLE IF NOT EXISTS drivers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    license_number  TEXT    UNIQUE,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    is_active       INTEGER NOT NULL DEFAULT 1  -- 1=active, 0=disabled
);

-- ── sessions ─────────────────────────────────
-- One row per driving session (camera-on period)
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id       INTEGER NOT NULL REFERENCES drivers(id) ON DELETE CASCADE,
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    duration_sec    INTEGER,                -- filled on session end
    avg_ear         REAL,                   -- average Eye Aspect Ratio
    total_alerts    INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

-- ── alerts ───────────────────────────────────
-- One row per drowsiness / microsleep event
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    alert_type      TEXT    NOT NULL CHECK(alert_type IN (
                        'eye_closure',   -- eyes closed > threshold
                        'microsleep',    -- brief sleep episode
                        'yawn',          -- yawn detected
                        'head_tilt'      -- head nodding down
                    )),
    severity        TEXT    NOT NULL DEFAULT 'medium' CHECK(severity IN ('low','medium','high')),
    ear_value       REAL,                   -- Eye Aspect Ratio at time of alert
    mar_value       REAL,                   -- Mouth Aspect Ratio (yawn)
    frame_timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
    acknowledged    INTEGER NOT NULL DEFAULT 0  -- 1 = driver responded
);

-- ── frame_logs ───────────────────────────────
-- Optional: per-frame metrics for analysis / model retraining
CREATE TABLE IF NOT EXISTS frame_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    captured_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    ear             REAL,
    mar             REAL,
    head_pitch      REAL,
    head_yaw        REAL,
    is_drowsy       INTEGER NOT NULL DEFAULT 0  -- model prediction label
);

-- ── config ───────────────────────────────────
-- Key-value store for tunable thresholds & app settings
CREATE TABLE IF NOT EXISTS config (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    description     TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── migrations ───────────────────────────────
-- Tracks which migration versions have been applied
CREATE TABLE IF NOT EXISTS migrations (
    version         INTEGER PRIMARY KEY,
    applied_at      TEXT NOT NULL DEFAULT (datetime('now')),
    description     TEXT
);
"""


# ──────────────────────────────────────────────
# INDEXES
# ──────────────────────────────────────────────
INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_sessions_driver   ON sessions(driver_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started  ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_alerts_session    ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_type       ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity   ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp  ON alerts(frame_timestamp);
CREATE INDEX IF NOT EXISTS idx_frame_logs_session ON frame_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_frame_logs_drowsy  ON frame_logs(is_drowsy);
"""


# ──────────────────────────────────────────────
# SEED DATA
# ──────────────────────────────────────────────
SEED_CONFIG = [
    # key,                  value,   description
    ("ear_threshold",       "0.25",  "Eye Aspect Ratio below this triggers eye_closure alert"),
    ("ear_consec_frames",   "20",    "Consecutive frames below EAR threshold before alert"),
    ("mar_threshold",       "0.75",  "Mouth Aspect Ratio above this triggers yawn alert"),
    ("alert_cooldown_sec",  "10",    "Seconds between repeated alerts of the same type"),
    ("alarm_sound_enabled", "1",     "Play audio alarm on alert (1=yes, 0=no)"),
    ("frame_log_enabled",   "1",     "Save per-frame metrics to frame_logs table (1=yes)"),
    ("fps_target",          "30",    "Target frames per second for camera capture"),
]

SEED_DRIVERS = [
    # name,          license_number
    ("Test Driver",  "TEST-001"),
]


def seed_data(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Insert config values (skip if key already exists)
    cur.executemany(
        """INSERT OR IGNORE INTO config (key, value, description)
           VALUES (?, ?, ?)""",
        SEED_CONFIG,
    )

    # Insert default test driver
    cur.executemany(
        """INSERT OR IGNORE INTO drivers (name, license_number)
           VALUES (?, ?)""",
        SEED_DRIVERS,
    )

    conn.commit()
    print(f"  ✔ Seeded {len(SEED_CONFIG)} config entries and {len(SEED_DRIVERS)} driver(s).")


# ──────────────────────────────────────────────
# MIGRATIONS
# ──────────────────────────────────────────────
# Add future schema changes here as new entries.
# Each entry: (version, description, sql_to_run)
MIGRATIONS = [
    (
        1,
        "Initial schema",
        None,   # version 1 IS the base schema — nothing extra to run
    ),
    # ── EXAMPLE future migration ──────────────────
    # (
    #     2,
    #     "Add face_confidence column to frame_logs",
    #     "ALTER TABLE frame_logs ADD COLUMN face_confidence REAL;"
    # ),
]


def run_migrations(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(version), 0) FROM migrations")
    current_version = cur.fetchone()[0]

    applied = 0
    for version, description, sql in MIGRATIONS:
        if version <= current_version:
            continue
        if sql:
            cur.executescript(sql)
        cur.execute(
            "INSERT INTO migrations (version, description) VALUES (?, ?)",
            (version, description),
        )
        conn.commit()
        print(f"  ✔ Migration v{version} applied: {description}")
        applied += 1

    if applied == 0:
        print("  ✔ Schema is up to date (no migrations needed).")


# ──────────────────────────────────────────────
# MAIN SETUP
# ──────────────────────────────────────────────
def setup_database(db_path: str = DB_PATH, force_reset: bool = False):
    """
    Create (or reset) the database.
    Set force_reset=True to wipe and rebuild from scratch (dev only).
    """
    if force_reset and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  ⚠  Existing database deleted: {db_path}")

    print(f"\n{'='*50}")
    print(f"  Driver Drowsiness DB Setup")
    print(f"  DB Path : {os.path.abspath(db_path)}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    conn = get_connection(db_path)

    print("► Creating tables...")
    conn.executescript(SCHEMA_SQL)
    print("  ✔ Tables ready.")

    print("► Creating indexes...")
    conn.executescript(INDEXES_SQL)
    print("  ✔ Indexes ready.")

    print("► Seeding initial data...")
    seed_data(conn)

    print("► Running migrations...")
    run_migrations(conn)

    conn.close()
    print(f"\n✅ Database setup complete → {db_path}\n")


# ──────────────────────────────────────────────
# QUICK VERIFY HELPER
# ──────────────────────────────────────────────
def verify_database(db_path: str = DB_PATH):
    """Print a summary of what's in the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    tables = ["drivers", "sessions", "alerts", "frame_logs", "config", "migrations"]
    print("\n── Database Verification ──")
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:<15} → {count} row(s)")

    print("\n── Config Values ──")
    for row in cur.execute("SELECT key, value, description FROM config"):
        print(f"  {row['key']:<25} = {row['value']:<8}  # {row['description']}")

    conn.close()
    print()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    setup_database()
    verify_database()