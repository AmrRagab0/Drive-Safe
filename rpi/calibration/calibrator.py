import time
import numpy as np

from detection.ear import compute_avg_ear
from detection.mar import compute_mar
from detection.head_pose import estimate_head_pose
from detection.blendshapes import eye_closure_score, jaw_open_score
from config import (
    CALIBRATION_DURATION,
    BLENDSHAPE_EYE_CLOSED_THRESHOLD,
    BLENDSHAPE_JAW_OPEN_THRESHOLD,
)


class Calibrator:
    """Per-driver calibration routine. Captures baseline metrics for 10 seconds."""

    def __init__(self, duration=CALIBRATION_DURATION):
        self.duration = duration

    def calibrate(self, camera, landmarker, global_start_time, fps=15):
        """Run calibration. Driver should look at camera naturally.

        Args:
            camera: Camera instance.
            landmarker: FaceLandmarker instance.
            global_start_time: Shared start time for monotonic timestamps.
            fps: Expected FPS for timestamp calculation.

        Returns:
            dict with calibrated thresholds, or None if calibration failed.
        """
        print(f"Calibration: Look at the camera naturally for {self.duration} seconds...")

        ear_samples = []
        mar_samples = []
        pitch_samples = []
        eye_closure_samples = []
        jaw_open_samples = []
        frame_count = 0
        start = time.time()

        while time.time() - start < self.duration:
            success, frame = camera.read()
            if not success:
                continue

            frame_count += 1
            timestamp_ms = int((time.time() - global_start_time) * 1000)

            # Convert BGR to RGB if needed (OpenCV captures BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                import cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            result = landmarker.detect(frame_rgb, timestamp_ms)
            if result is None:
                continue

            # EAR
            left_eye, right_eye = landmarker.get_eye_landmarks(result, frame.shape)
            ear = compute_avg_ear(left_eye, right_eye)
            ear_samples.append(ear)

            # MAR
            mouth = landmarker.get_mouth_landmarks(result, frame.shape)
            mar = compute_mar(mouth)
            mar_samples.append(mar)

            # Head pose
            head_points = landmarker.get_head_pose_landmarks(result, frame.shape)
            pitch, _, _ = estimate_head_pose(head_points, frame.shape)
            pitch_samples.append(pitch)

            # Blendshapes
            blendshapes = landmarker.get_blendshapes(result)
            if blendshapes is not None:
                eye_closure_samples.append(eye_closure_score(blendshapes))
                jaw_open_samples.append(jaw_open_score(blendshapes))

        if len(ear_samples) < 10:
            print("Calibration failed: not enough face detections.")
            return None

        ear_mean = np.mean(ear_samples)
        ear_std = np.std(ear_samples)
        mar_mean = np.mean(mar_samples)
        mar_std = np.std(mar_samples)

        thresholds = {
            "ear_closed": max(ear_mean - 2 * ear_std, ear_mean * 0.6),
            "mar_yawn": mar_mean + 3 * mar_std,
            "pitch_neutral": np.mean(pitch_samples),
        }

        print(f"Calibration complete:")
        print(f"  EAR baseline: {ear_mean:.3f} (threshold: {thresholds['ear_closed']:.3f})")
        print(f"  MAR baseline: {mar_mean:.3f} (yawn threshold: {thresholds['mar_yawn']:.3f})")
        print(f"  Pitch neutral: {thresholds['pitch_neutral']:.1f} degrees")

        # Blendshape thresholds: blink/jaw scores are bimodal — near 0 when
        # "open"/"closed" baseline, near 1 when triggered. A fixed 0.5 midpoint
        # works for most people; we floor the calibrated threshold at the
        # defaults so a noisy baseline can't push it too low/high.
        if eye_closure_samples:
            eye_mean = float(np.mean(eye_closure_samples))
            eye_std = float(np.std(eye_closure_samples))
            jaw_mean = float(np.mean(jaw_open_samples))
            jaw_std = float(np.std(jaw_open_samples))

            blendshape_eye_closed = max(
                BLENDSHAPE_EYE_CLOSED_THRESHOLD,
                eye_mean + 5 * eye_std,
            )
            blendshape_jaw_open = max(
                BLENDSHAPE_JAW_OPEN_THRESHOLD,
                jaw_mean + 5 * jaw_std,
            )
            thresholds["blendshape_eye_closed"] = blendshape_eye_closed
            thresholds["blendshape_jaw_open"] = blendshape_jaw_open

            print(f"  eyeBlink baseline: {eye_mean:.3f} ± {eye_std:.3f} "
                  f"(closed threshold: {blendshape_eye_closed:.3f})")
            print(f"  jawOpen baseline:  {jaw_mean:.3f} ± {jaw_std:.3f} "
                  f"(yawn threshold:    {blendshape_jaw_open:.3f})")

        return thresholds
