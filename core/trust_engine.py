import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os

class TrustEngine:
    """
    Calculates the Driver Trust Index at the end of a session
    and updates their historical sensitivity modifier.
    """
    def __init__(self, db_path="drowsiness.db"):
        self.db_path = db_path
        
        # Penalty weights: Mild, High, Critical
        self.weights = {1: 2.0, 2: 6.0, 3: 15.0}

    def calculate_session_trust(self, driver_id: int, session_start: datetime, session_end: datetime) -> float:
        """Calculates the 0-100 Trust Score for a specific session."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Pull events for this session timeframe
            query = """
                SELECT fatigue_level 
                FROM fatigue_events 
                WHERE driver_id = ? AND timestamp BETWEEN ? AND ?
                AND fatigue_level > 0
            """
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(driver_id, session_start.isoformat(), session_end.isoformat())
            )
            conn.close()

            # 1. Calculate Session Duration in Hours
            duration_hours = (session_end - session_start).total_seconds() / 3600.0
            duration_factor = np.sqrt(max(0.5, duration_hours))

            # 2. Calculate Event Penalties
            if df.empty:
                return 100.0  # Perfect drive
                
            event_counts = df['fatigue_level'].value_counts().to_dict()
            
            c1 = event_counts.get(1, 0)
            c2 = event_counts.get(2, 0)
            c3 = event_counts.get(3, 0)
            
            total_penalty = (c1 * self.weights[1]) + (c2 * self.weights[2]) + (c3 * self.weights[3])

            # 3. Apply Formula
            trust_score = 100.0 - (total_penalty / duration_factor)
            
            return round(max(0.0, trust_score), 1)

        except Exception as e:
            print(f"[TRUST ENGINE] Error calculating score: {e}")
            return 100.0

    def update_driver_profile(self, driver_id: int, new_session_trust: float):
        """
        Updates the driver's global Trust Index and adjusts the AI sensitivity.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Fetch current profile
        cur.execute("SELECT trust_index FROM driver_profiles WHERE driver_id = ?", (driver_id,))
        row = cur.fetchone()
        
        if row:
            historical_trust = row[0]
            # EMA for global trust: 70% history, 30% new session
            updated_trust = (historical_trust * 0.7) + (new_session_trust * 0.3)
        else:
            updated_trust = new_session_trust

        # Determine new AI sensitivity (Lower score = higher sensitivity)
        # Baseline 1.0. Drops to 0.85 (chill) or spikes to 1.3 (strict)
        if updated_trust >= 90:
            new_sensitivity = 0.85 
        elif updated_trust >= 75:
            new_sensitivity = 1.0  
        elif updated_trust >= 50:
            new_sensitivity = 1.15 
        else:
            new_sensitivity = 1.3  

        # Save to DB
        cur.execute("""
            UPDATE driver_profiles 
            SET trust_index = ?, sensitivity_modifier = ?, updated_at = ?
            WHERE driver_id = ?
        """, (round(updated_trust, 1), new_sensitivity, datetime.now().isoformat(), driver_id))
        
        conn.commit()
        conn.close()
        
        print(f"[TRUST ENGINE] Driver {driver_id} updated. New Trust: {updated_trust:.1f} | Sensitivity: {new_sensitivity}")
        return updated_trust