import numpy as np
import cv2

# Generic 3D face model points (in arbitrary units)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)


def estimate_head_pose(image_points, frame_shape):
    """Estimate head pitch, yaw, roll from 6 facial landmarks.

    Args:
        image_points: numpy array of shape (6, 2) — 2D image coordinates.
        frame_shape: (height, width, ...) of the frame.

    Returns:
        (pitch, yaw, roll) in degrees.
    """
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2.0, h / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll
