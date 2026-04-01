import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from config import (
    MEDIAPIPE_MODEL_PATH,
    MIN_FACE_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

# MediaPipe face mesh landmark indices
# Eyes: using the 6-point EAR model per eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth: 8 landmarks for MAR
UPPER_LIP = [61, 39, 0, 269]     # left corner, upper-left, top, upper-right
LOWER_LIP = [291, 181, 17, 405]  # right corner, lower-left, bottom, lower-right
MOUTH = [61, 39, 0, 269, 291, 405, 17, 181]
# Ordered: left-corner, upper-left, top, upper-right, right-corner, lower-right, bottom, lower-left

# Head pose: 6 key landmarks
HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 291, 61]
# nose tip, chin, left eye corner, right eye corner, left mouth, right mouth


class FaceLandmarker:
    """Wrapper around MediaPipe FaceLandmarker for drowsiness detection."""

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb, timestamp_ms):
        """Detect face landmarks in an RGB frame.

        Args:
            frame_rgb: RGB image as numpy array.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            FaceLandmarkerResult or None if no face detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None
        return result

    def get_eye_landmarks(self, result, frame_shape):
        """Extract left and right eye landmarks as numpy arrays."""
        landmarks = result.face_landmarks[0]
        h, w = frame_shape[:2]

        left_eye = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE],
            dtype=np.float64,
        )
        right_eye = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE],
            dtype=np.float64,
        )
        return left_eye, right_eye

    def get_mouth_landmarks(self, result, frame_shape):
        """Extract mouth landmarks as numpy array (8 points)."""
        landmarks = result.face_landmarks[0]
        h, w = frame_shape[:2]

        mouth = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH],
            dtype=np.float64,
        )
        return mouth

    def get_head_pose_landmarks(self, result, frame_shape):
        """Extract 6 key landmarks for head pose estimation as 2D image points."""
        landmarks = result.face_landmarks[0]
        h, w = frame_shape[:2]

        points = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )
        return points

    def get_blendshapes(self, result):
        """Extract blendshape scores as a dict. Returns None if unavailable."""
        if not result.face_blendshapes:
            return None

        return {
            bs.category_name: bs.score
            for bs in result.face_blendshapes[0]
        }
