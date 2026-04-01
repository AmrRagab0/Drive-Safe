import time
from collections import deque

from config import (
    PERCLOS_WINDOW_SECONDS,
    BLINK_DROWSY_THRESHOLD,
    YAWN_MIN_DURATION,
    YAWN_WINDOW,
    YAWN_COUNT_THRESHOLD,
    HEAD_NOD_PITCH_DROP,
    HEAD_YAW_THRESHOLD,
    HEAD_POSE_DURATION,
)


class DrowsinessDetector:
    """Tracks all drowsiness metrics over time."""

    def __init__(self, fps=15):
        self.fps = fps
        self.ear_threshold = 0.20  # overridden by calibration

        # PERCLOS
        self.perclos_window = deque(maxlen=fps * PERCLOS_WINDOW_SECONDS)

        # Blink tracking
        self.blink_durations = deque(maxlen=100)
        self._blink_start = None
        self._eyes_closed_since = None

        # Yawn tracking
        self.yawn_timestamps = deque()
        self._yawn_start = None
        self._yawning = False
        self.mar_threshold = 0.6  # overridden by calibration

        # Head pose tracking
        self.pitch_neutral = 0.0  # overridden by calibration
        self.pitch_nod_threshold = -15.0
        self._head_nod_start = None
        self._looking_away_start = None

    # --- PERCLOS & Blink ---

    def update_ear(self, ear):
        """Update PERCLOS and blink tracking with a new EAR reading.

        Returns:
            perclos (float): Current PERCLOS value (0.0 - 1.0).
        """
        now = time.time()
        is_closed = ear < self.ear_threshold
        self.perclos_window.append(1 if is_closed else 0)

        # Track eye closure duration
        if is_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            if self._blink_start is None:
                self._blink_start = now
        else:
            if self._blink_start is not None:
                duration = now - self._blink_start
                if duration > 0.05:  # ignore noise < 50ms
                    self.blink_durations.append(duration)
                self._blink_start = None
            self._eyes_closed_since = None

        if not self.perclos_window:
            return 0.0
        return sum(self.perclos_window) / len(self.perclos_window)

    def get_avg_blink_duration(self, last_n=10):
        """Average blink duration over the last N blinks."""
        recent = list(self.blink_durations)[-last_n:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def current_closure_duration(self):
        """How long eyes have been continuously closed (seconds)."""
        if self._eyes_closed_since is None:
            return 0.0
        return time.time() - self._eyes_closed_since

    # --- Yawn ---

    def update_mar(self, mar):
        """Update yawn tracking with a new MAR reading.

        Returns:
            True if a yawn was just completed.
        """
        now = time.time()
        is_open = mar > self.mar_threshold

        yawn_completed = False

        if is_open:
            if self._yawn_start is None:
                self._yawn_start = now
            elif not self._yawning and (now - self._yawn_start) >= YAWN_MIN_DURATION:
                self._yawning = True
        else:
            if self._yawning:
                self.yawn_timestamps.append(now)
                yawn_completed = True
            self._yawn_start = None
            self._yawning = False

        # Clean old yawns outside the window
        cutoff = now - YAWN_WINDOW
        while self.yawn_timestamps and self.yawn_timestamps[0] < cutoff:
            self.yawn_timestamps.popleft()

        return yawn_completed

    @property
    def yawn_count_5min(self):
        """Number of yawns in the last 5 minutes."""
        return len(self.yawn_timestamps)

    # --- Head Pose ---

    def check_head_nod(self, pitch):
        """Check if driver is nodding (pitch drop below threshold).

        Returns:
            True if head nod is detected.
        """
        threshold = self.pitch_neutral + self.pitch_nod_threshold
        is_nodding = pitch < threshold

        now = time.time()
        if is_nodding:
            if self._head_nod_start is None:
                self._head_nod_start = now
            return True
        else:
            self._head_nod_start = None
            return False

    def check_looking_away(self, yaw):
        """Check if driver is looking away for too long.

        Returns:
            True if looking away for longer than HEAD_POSE_DURATION.
        """
        now = time.time()
        if abs(yaw) > HEAD_YAW_THRESHOLD:
            if self._looking_away_start is None:
                self._looking_away_start = now
            return (now - self._looking_away_start) >= HEAD_POSE_DURATION
        else:
            self._looking_away_start = None
            return False

    def apply_calibration(self, thresholds):
        """Apply calibrated thresholds."""
        self.ear_threshold = thresholds["ear_closed"]
        self.mar_threshold = thresholds["mar_yawn"]
        self.pitch_neutral = thresholds["pitch_neutral"]
        self.pitch_nod_threshold = -HEAD_NOD_PITCH_DROP
