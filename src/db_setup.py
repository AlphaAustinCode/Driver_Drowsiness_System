import sqlite3
import os
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DB_PATH        = "drowsiness.db"
SCHEMA_VERSION = 3  # Bumped for Day 4 additions


# ──────────────────────────────────────────────
# CONNECTION HELPER
# ──────────────────────────────────────────────
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ──────────────────────────────────────────────
# SCHEMA
# ──────────────────────────────────────────────
SCHEMA_SQL = """
-- ── drivers ──────────────────────────────────
CREATE TABLE IF NOT EXISTS drivers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    license_number  TEXT    UNIQUE,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    is_active       INTEGER NOT NULL DEFAULT 1
);

-- ── sessions ─────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id       INTEGER NOT NULL REFERENCES drivers(id) ON DELETE CASCADE,
    start_time      TEXT    NOT NULL DEFAULT (datetime('now')),
    end_time        TEXT,
    status          TEXT    NOT NULL DEFAULT 'active',
    total_alerts    INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

-- ── alerts ───────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    alert_type      TEXT    NOT NULL,
    ear_value       REAL,
    mar_value       REAL,
    cnn_class       TEXT,
    cnn_confidence  REAL
);

-- ── frame_logs ───────────────────────────────
CREATE TABLE IF NOT EXISTS frame_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    ear_value       REAL,
    mar_value       REAL,
    cnn_class       TEXT,
    cnn_confidence  REAL,
    is_drowsy       INTEGER NOT NULL DEFAULT 0
);

-- ── driver_profiles (Day 4) ──────────────────
CREATE TABLE IF NOT EXISTS driver_profiles (
    driver_id             INTEGER PRIMARY KEY REFERENCES drivers(id) ON DELETE CASCADE,
    sensitivity_modifier  REAL NOT NULL DEFAULT 1.0,
    trust_index           REAL NOT NULL DEFAULT 100.0,
    updated_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── fatigue_events (Day 4) ───────────────────
CREATE TABLE IF NOT EXISTS fatigue_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id        INTEGER NOT NULL REFERENCES drivers(id) ON DELETE CASCADE,
    fatigue_level    INTEGER NOT NULL,
    fatigue_score    REAL NOT NULL,
    circadian_label  TEXT,
    timestamp        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── config ───────────────────────────────────
CREATE TABLE IF NOT EXISTS config (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    description     TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── migrations ───────────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_sessions_driver    ON sessions(driver_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start     ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_alerts_session     ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_type        ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp   ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_frame_logs_session ON frame_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_frame_logs_drowsy  ON frame_logs(is_drowsy);
CREATE INDEX IF NOT EXISTS idx_fatigue_driver     ON fatigue_events(driver_id);
"""


# ──────────────────────────────────────────────
# SEED DATA
# ──────────────────────────────────────────────
SEED_CONFIG = [
    ("ear_threshold",       "0.25",  "Eye Aspect Ratio below this triggers eye_closure alert"),
    ("ear_consec_frames",   "20",    "Consecutive frames below EAR threshold before alert"),
    ("mar_threshold",       "0.60",  "Mouth Aspect Ratio above this triggers yawn alert"),
    ("cnn_threshold",       "0.70",  "Minimum CNN confidence to act on prediction"),
    ("alert_cooldown_sec",  "4",     "Seconds between repeated alerts"),
    ("alarm_sound_enabled", "1",     "Play audio alarm on alert (1=yes, 0=no)"),
    ("frame_log_enabled",   "1",     "Save per-frame metrics to frame_logs (1=yes)"),
    ("fps_target",          "30",    "Target frames per second for camera capture"),
]

SEED_DRIVERS = [
    ("Default Driver", "DEFAULT-001"),
]


def seed_data(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO config (key, value, description) VALUES (?, ?, ?)",
        SEED_CONFIG,
    )
    cur.executemany(
        "INSERT OR IGNORE INTO drivers (name, license_number) VALUES (?, ?)",
        SEED_DRIVERS,
    )
    conn.commit()
    print(f"  + Seeded {len(SEED_CONFIG)} config entries and {len(SEED_DRIVERS)} driver(s).")


# ──────────────────────────────────────────────
# MIGRATIONS
# ──────────────────────────────────────────────
MIGRATIONS = [
    (1, "Initial schema (Day 1)", None),
    (2, "Day 3 schema — aligned column names for real-time detection", None),
    (3, "Day 4 schema — added driver_profiles and fatigue_events", """
        INSERT OR IGNORE INTO driver_profiles (driver_id, sensitivity_modifier) 
        SELECT id, 1.0 FROM drivers;
    """),
]


def run_migrations(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Check if migrations table exists before querying
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'")
    if not cur.fetchone():
        return # Setup script will handle initial table creation

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
        print(f"  + Migration v{version} applied: {description}")
        applied += 1

    if applied == 0:
        print("  + Schema is up to date.")


# ──────────────────────────────────────────────
# MAIN SETUP
# ──────────────────────────────────────────────
def setup_database(db_path: str = DB_PATH, force_reset: bool = False):
    if force_reset and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Existing database deleted: {db_path}")

    print(f"\n{'='*50}")
    print(f"  Driver Drowsiness DB Setup (Day 4)")
    print(f"  DB Path : {os.path.abspath(db_path)}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    conn = get_connection(db_path)

    print("Creating tables...")
    conn.executescript(SCHEMA_SQL)
    print("  + Tables ready.")

    print("Creating indexes...")
    conn.executescript(INDEXES_SQL)
    print("  + Indexes ready.")

    print("Seeding initial data...")
    seed_data(conn)

    print("Running migrations...")
    run_migrations(conn)

    conn.close()
    print(f"\nDatabase setup complete -> {db_path}\n")


# ──────────────────────────────────────────────
# VERIFY
# ──────────────────────────────────────────────
def verify_database(db_path: str = DB_PATH):
    conn = get_connection(db_path)
    cur  = conn.cursor()

    # Updated list to include new Day 4 tables
    tables = ["drivers", "sessions", "alerts", "frame_logs", "driver_profiles", "fatigue_events", "config", "migrations"]
    print("\n-- Database Verification --")
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table:<20} -> {count} row(s)")
        except sqlite3.OperationalError:
            print(f"  {table:<20} -> NOT FOUND")

    print("\n-- Config Values --")
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