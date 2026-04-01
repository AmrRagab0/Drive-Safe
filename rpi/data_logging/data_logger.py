import sqlite3
import time


class DataLogger:
    """SQLite logger for drowsiness detection metrics."""

    def __init__(self, db_path="drowsiness_log.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                timestamp REAL,
                perclos REAL,
                ear REAL,
                mar REAL,
                pitch REAL,
                yaw REAL,
                roll REAL,
                state TEXT,
                score REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                timestamp REAL,
                event TEXT,
                details TEXT
            )
        """)
        self.conn.commit()
        self._batch = []
        self._batch_size = 30  # commit every 30 readings (~1-2 seconds)

    def log(self, perclos, ear, mar, pitch, yaw, roll, state, score):
        """Log a single frame's metrics."""
        self._batch.append((
            time.time(), perclos, ear, mar, pitch, yaw, roll, state.name, score
        ))
        if len(self._batch) >= self._batch_size:
            self.flush()

    def log_event(self, event, details=""):
        """Log a discrete event (e.g., FACE_LOST, CALIBRATION_DONE)."""
        self.conn.execute(
            "INSERT INTO events VALUES (?, ?, ?)",
            (time.time(), event, details),
        )
        self.conn.commit()

    def flush(self):
        """Write pending readings to disk."""
        if self._batch:
            self.conn.executemany(
                "INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._batch,
            )
            self.conn.commit()
            self._batch.clear()

    def close(self):
        self.flush()
        self.conn.close()
