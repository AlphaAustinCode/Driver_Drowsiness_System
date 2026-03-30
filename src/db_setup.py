import sqlite3
import os
from datetime import datetime

DB_PATH = "drowsiness.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    print(f"\n[DB] Initializing Day 4 Schema...")
    conn = get_connection()
    cur = conn.cursor()

    # 1. Create Core Tables (If they don't exist)
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS drivers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        license_number TEXT UNIQUE
    );

    CREATE TABLE IF NOT EXISTS driver_profiles (
        driver_id INTEGER PRIMARY KEY REFERENCES drivers(id)
    );

    CREATE TABLE IF NOT EXISTS fatigue_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id INTEGER NOT NULL REFERENCES drivers(id),
        fatigue_level INTEGER,
        fatigue_score REAL,
        circadian_label TEXT,
        timestamp TEXT DEFAULT (datetime('now'))
    );
    """)

    # 2. FORCE PATCH: Manually add columns to existing driver_profiles
    # This prevents the "Table Exists but Column Missing" error
    column_patches = [
        ("trust_index", "REAL DEFAULT 100.0"),
        ("sensitivity_modifier", "REAL DEFAULT 1.0"),
        ("updated_at", "TEXT DEFAULT (datetime('now'))")
    ]

    for col_name, col_type in column_patches:
        try:
            cur.execute(f"ALTER TABLE driver_profiles ADD COLUMN {col_name} {col_type}")
            print(f"  + Added missing column: {col_name}")
        except sqlite3.OperationalError:
            # Column already exists, which is fine
            pass

    # 3. Seed Default Driver
    cur.execute("INSERT OR IGNORE INTO drivers (id, name, license_number) VALUES (1, 'Default Driver', 'DEF-001')")
    cur.execute("INSERT OR IGNORE INTO driver_profiles (driver_id) VALUES (1)")

    conn.commit()
    conn.close()
    print("[DB] Setup and Patching Complete.")

def verify_database():
    conn = get_connection()
    cur = conn.cursor()
    print("\n--Final Integrity Check --")
    
    # Verify Columns in driver_profiles
    cur.execute("PRAGMA table_info(driver_profiles)")
    columns = [row[1] for row in cur.fetchall()]
    
    check_cols = ["trust_index", "sensitivity_modifier"]
    all_passed = True
    
    for c in check_cols:
        status = "OK ✅" if c in columns else "MISSING ❌"
        if "MISSING" in status: all_passed = False
        print(f"  Column {c:<22} : {status}")

    if all_passed:
        print("\n[SUCCESS] Your database is 100% ready for the Trust Engine.")
    else:
        print("\n[ERROR] Database mismatch detected.")
    
    conn.close()

if __name__ == "__main__":
    setup_database()
    verify_database()