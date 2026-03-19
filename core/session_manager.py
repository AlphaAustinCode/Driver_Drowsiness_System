"""
core/session_manager.py
=======================
Day 3 — Session Manager
Driver Drowsiness Detection System

Handles session lifecycle: start, frame logging throttle, end.
Wraps db_queries so detector/alert code stays clean.

Author: Austin Trinidad
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_queries import (
    get_or_create_driver, start_session, end_session, log_frame
)


class SessionManager:
    """
    Manages a single driving session lifecycle.

    Usage:
        sm = SessionManager()
        sm.begin()
        sm.record_frame(ear, mar, cnn_class, cnn_confidence, is_drowsy)
        sm.finish(total_alerts)
    """

    # Log one frame to DB every N frames — prevents flooding the DB
    FRAME_LOG_INTERVAL = 15

    def __init__(self, driver_name: str = "Default Driver"):
        self.driver_name  = driver_name
        self.driver_id    = None
        self.session_id   = None
        self.frame_count  = 0
        self.active       = False

    def begin(self):
        """Start a new session and record to DB."""
        self.driver_id  = get_or_create_driver(self.driver_name)
        self.session_id = start_session(self.driver_id)
        self.frame_count = 0
        self.active = True
        print(f"[SESSION] Started — driver: '{self.driver_name}' | session_id: {self.session_id}")

    def record_frame(self, ear: float, mar: float,
                     cnn_class: str, cnn_confidence: float, is_drowsy: bool):
        """
        Log frame data to DB at throttled rate.
        Called every frame but only writes every FRAME_LOG_INTERVAL frames.
        """
        if not self.active or self.session_id is None:
            return

        self.frame_count += 1

        if self.frame_count % self.FRAME_LOG_INTERVAL == 0:
            log_frame(
                session_id=self.session_id,
                ear=ear,
                mar=mar,
                cnn_class=cnn_class,
                cnn_confidence=cnn_confidence,
                is_drowsy=is_drowsy
            )

    def finish(self, total_alerts: int):
        """End the session and write summary to DB."""
        if not self.active or self.session_id is None:
            return

        end_session(self.session_id, total_alerts)
        self.active = False
        print(f"[SESSION] Ended — total frames: {self.frame_count} | alerts: {total_alerts}")