import time
from enum import IntEnum

from config import RECOVERY_TIME


class DrowsinessState(IntEnum):
    ALERT = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class StateMachine:
    """4-level drowsiness state machine with hysteresis."""

    def __init__(self):
        self.state = DrowsinessState.ALERT
        self.state_entry_time = time.time()

    def update(self, perclos, avg_blink_dur, yawn_count_5min, head_nod, eyes_closed_duration):
        """Update state based on current metrics.

        Returns:
            Current DrowsinessState.
        """
        now = time.time()
        time_in_state = now - self.state_entry_time

        if self.state == DrowsinessState.ALERT:
            if perclos > 0.15 or yawn_count_5min >= 3:
                self._transition(DrowsinessState.MILD)

        elif self.state == DrowsinessState.MILD:
            active_signals = sum([
                perclos > 0.15,
                avg_blink_dur > 0.5,
                yawn_count_5min >= 3,
                head_nod,
            ])
            if active_signals >= 2 or perclos > 0.20:
                self._transition(DrowsinessState.MODERATE)
            elif perclos < 0.10 and not head_nod and yawn_count_5min < 2:
                if time_in_state > RECOVERY_TIME:
                    self._transition(DrowsinessState.ALERT)

        elif self.state == DrowsinessState.MODERATE:
            if perclos > 0.30 or head_nod or eyes_closed_duration > 2.0:
                self._transition(DrowsinessState.SEVERE)
            elif perclos < 0.10 and not head_nod:
                if time_in_state > RECOVERY_TIME:
                    self._transition(DrowsinessState.ALERT)

        elif self.state == DrowsinessState.SEVERE:
            if perclos < 0.10 and not head_nod and eyes_closed_duration < 0.5:
                if time_in_state > RECOVERY_TIME:
                    self._transition(DrowsinessState.MODERATE)

        return self.state

    def _transition(self, new_state):
        self.state = new_state
        self.state_entry_time = time.time()

    def reset(self):
        self.state = DrowsinessState.ALERT
        self.state_entry_time = time.time()
