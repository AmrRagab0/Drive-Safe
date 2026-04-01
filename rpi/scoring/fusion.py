from config import (
    WEIGHT_PERCLOS,
    WEIGHT_BLINK,
    WEIGHT_YAWN,
    WEIGHT_HEAD_POSE,
    WEIGHT_GAZE,
    PERCLOS_MILD_THRESHOLD,
    BLINK_DROWSY_THRESHOLD,
    YAWN_COUNT_THRESHOLD,
)


def compute_drowsiness_score(perclos, avg_blink_dur, yawn_count_5min, head_nod, looking_away):
    """Compute weighted drowsiness score (0.0 = alert, 1.0 = very drowsy).

    Each signal is normalized to 0.0-1.0 range before weighting.
    """
    # Normalize PERCLOS: 0 at threshold, 1.0 at 2x threshold
    perclos_norm = min(perclos / (PERCLOS_MILD_THRESHOLD * 2), 1.0)

    # Normalize blink duration: 0 below threshold, 1.0 at 2x threshold
    blink_norm = min(max(avg_blink_dur - BLINK_DROWSY_THRESHOLD, 0) / BLINK_DROWSY_THRESHOLD, 1.0)

    # Normalize yawn count: 0 at 0, 1.0 at threshold
    yawn_norm = min(yawn_count_5min / YAWN_COUNT_THRESHOLD, 1.0)

    # Head pose: binary signals
    head_norm = 1.0 if head_nod else 0.0
    gaze_norm = 1.0 if looking_away else 0.0

    score = (
        WEIGHT_PERCLOS * perclos_norm
        + WEIGHT_BLINK * blink_norm
        + WEIGHT_YAWN * yawn_norm
        + WEIGHT_HEAD_POSE * head_norm
        + WEIGHT_GAZE * gaze_norm
    )

    return min(score, 1.0)
