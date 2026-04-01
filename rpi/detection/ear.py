import numpy as np


def compute_ear(eye_landmarks):
    """Compute Eye Aspect Ratio from 6 landmark points.

    Points are ordered: [outer corner, upper-1, upper-2, inner corner, lower-1, lower-2]
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Args:
        eye_landmarks: numpy array of shape (6, 2).

    Returns:
        EAR value (float). ~0.25-0.30 when open, <0.20 when closed.
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_avg_ear(left_eye, right_eye):
    """Average EAR across both eyes."""
    return (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
