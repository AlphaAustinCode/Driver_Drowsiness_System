"""
migrate_day4.py
===============
Day 4 — Database Migration
Driver Drowsiness Detection System

Adds two new tables required by fatigue_classifier.py:

  driver_profiles
    - Stores per-driver sensitivity modifier (from Trust Index)
    - sensitivity_modifier > 1.0 → driver tends to ignore alerts → more sensitive
    - sensitivity_modifier < 1.0 → driver has good history → slightly relaxed

  fatigue_events
    - Logs fatigue level transitions per session
    - Used by dashboard to show fatigue timeline
    - Only written when level CHANGES (not every frame)

Run this ONCE before starting Day 4 detection:
    python migrate_day4.py

Safe to run multiple times — uses IF NOT EXISTS.

Author: Austin Trinidad
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness.db")

MIGRATION_SQL = """
-- ── driver_profiles ──────────────────────────────────────────────────────────
-- Stores adaptive sensitivity modifier per driver.
-- sensitivity_modifier is updated by Trust Index (Step 10) after each session.
CREATE TABLE IF NOT EXISTS driver_profiles (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id            INTEGER NOT NULL UNIQUE REFERENCES drivers(id) ON DELETE CASCADE,
    sensitivity_modifier REAL    NOT NULL DEFAULT 1.0,
    avg_fatigue_score    REAL    NOT NULL DEFAULT 0.0,
    total_sessions       INTEGER NOT NULL DEFAULT 0,
    total_alerts         INTEGER NOT NULL DEFAULT 0,
    total_ignored_alerts INTEGER NOT NULL DEFAULT 0,
    last_updated         TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── fatigue_events ────────────────────────────────────────────────────────────
-- Logs fatigue level transitions (Level 0→1, 1→2, etc.) during a session.
-- Written only when level changes — not every frame.
-- Powers the fatigue timeline chart in the Day 5 dashboard.
CREATE TABLE IF NOT EXISTS fatigue_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id       INTEGER NOT NULL REFERENCES drivers(id) ON DELETE CASCADE,
    fatigue_level   INTEGER NOT NULL,
    fatigue_score   REAL    NOT NULL,
    circadian_label TEXT    NOT NULL DEFAULT 'low',
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_driver_profiles_driver ON driver_profiles(driver_id);
CREATE INDEX IF NOT EXISTS idx_fatigue_events_driver  ON fatigue_events(driver_id);
CREATE INDEX IF NOT EXISTS idx_fatigue_events_ts      ON fatigue_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_fatigue_events_level   ON fatigue_events(fatigue_level);
"""

SEED_PROFILE_SQL = """
INSERT OR IGNORE INTO driver_profiles (driver_id, sensitivity_modifier)
SELECT id, 1.0 FROM drivers;
"""


def run_migration():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        print("Run db_setup.py first.")
        return

    print(f"\n{'='*55}")
    print(f"  Day 4 DB Migration")
    print(f"  DB   : {os.path.abspath(DB_PATH)}")
    print(f"  Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    print("Adding new tables...")
    conn.executescript(MIGRATION_SQL)
    print("  + driver_profiles table ready.")
    print("  + fatigue_events table ready.")

    print("Adding indexes...")
    conn.executescript(INDEXES_SQL)
    print("  + Indexes ready.")

    print("Seeding driver profiles for existing drivers...")
    conn.executescript(SEED_PROFILE_SQL)

    # Count seeded profiles
    count = conn.execute("SELECT COUNT(*) FROM driver_profiles").fetchone()[0]
    print(f"  + {count} driver profile(s) seeded.")

    # Log migration in migrations table
    current = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM migrations"
    ).fetchone()[0]

    if current < 3:
        conn.execute(
            "INSERT OR IGNORE INTO migrations (version, description) VALUES (?, ?)",
            (3, "Day 4 — Added driver_profiles and fatigue_events tables")
        )
        conn.commit()
        print("  + Migration v3 recorded.")
    else:
        print("  + Migration already recorded.")

    conn.close()
    print(f"\nMigration complete.\n")

    # Verify
    print("-- Verification --")
    conn = sqlite3.connect(DB_PATH)
    tables = ["driver_profiles", "fatigue_events"]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<25} → {count} row(s)")
    conn.close()
    print()


if __name__ == "__main__":
    run_migration()