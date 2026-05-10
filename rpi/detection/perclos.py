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
    BLENDSHAPE_EYE_CLOSED_THRESHOLD,
    BLENDSHAPE_JAW_OPEN_THRESHOLD,
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
        self.blendshape_eye_closed_threshold = BLENDSHAPE_EYE_CLOSED_THRESHOLD
        self.blendshape_jaw_open_threshold = BLENDSHAPE_JAW_OPEN_THRESHOLD

        # Head pose tracking
        self.pitch_neutral = 0.0  # overridden by calibration
        self.pitch_nod_threshold = -15.0
        self._head_nod_start = None
        self._looking_away_start = None

    # --- PERCLOS & Blink ---

    def update_eye_closed(self, is_closed):
        """Update PERCLOS and blink tracking from a per-frame closed/open boolean.

        Lets callers choose the signal (EAR threshold, blendshape threshold, or
        a learned classifier) without coupling this module to any one of them.

        Returns:
            perclos (float): Current PERCLOS value (0.0 - 1.0).
        """
        now = time.time()
        self.perclos_window.append(1 if is_closed else 0)

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

    def update_ear(self, ear):
        """EAR-based wrapper around update_eye_closed."""
        return self.update_eye_closed(ear < self.ear_threshold)

    def update_eye_closure_score(self, closure_score):
        """Blendshape-based wrapper around update_eye_closed.

        Args:
            closure_score: max(eyeBlinkLeft, eyeBlinkRight) in [0, 1].
        """
        return self.update_eye_closed(closure_score > self.blendshape_eye_closed_threshold)

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

    def update_mouth_open(self, is_open):
        """Update yawn tracking from a per-frame mouth-open boolean.

        Returns:
            True if a yawn was just completed.
        """
        now = time.time()
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

    def update_mar(self, mar):
        """MAR-based wrapper around update_mouth_open."""
        return self.update_mouth_open(mar > self.mar_threshold)

    def update_jaw_open_score(self, jaw_open_score):
        """Blendshape-based wrapper around update_mouth_open.

        Args:
            jaw_open_score: jawOpen blendshape in [0, 1].
        """
        return self.update_mouth_open(jaw_open_score > self.blendshape_jaw_open_threshold)

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
        if "blendshape_eye_closed" in thresholds:
            self.blendshape_eye_closed_threshold = thresholds["blendshape_eye_closed"]
        if "blendshape_jaw_open" in thresholds:
            self.blendshape_jaw_open_threshold = thresholds["blendshape_jaw_open"]
