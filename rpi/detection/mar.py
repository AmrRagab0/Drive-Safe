import numpy as np


def compute_mar(mouth_landmarks):
    """Compute Mouth Aspect Ratio from 8 mouth landmarks.

    Points ordered: [left-corner, upper-left, top, upper-right,
                     right-corner, lower-right, bottom, lower-left]
    MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|)

    Args:
        mouth_landmarks: numpy array of shape (8, 2).

    Returns:
        MAR value (float). ~0.1-0.3 normal, >0.6 yawning.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_landmarks

    vertical_1 = np.linalg.norm(p2 - p8)
    vertical_2 = np.linalg.norm(p3 - p7)
    vertical_3 = np.linalg.norm(p4 - p6)
    horizontal = np.linalg.norm(p1 - p5)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
