#!/usr/bin/env python3
"""Drive Safe 2.0 — Driver Drowsiness Detection System.

Usage:
    python main.py                      # Default webcam (index 0)
    python main.py --source 0           # Webcam / DroidCam
    python main.py --source video.mp4   # Pre-recorded video
    python main.py --source picamera2   # RPi native camera
    python main.py --no-calibration     # Skip calibration (use defaults)
    python main.py --no-display         # Headless mode (no OpenCV window)
"""

import argparse
import sys
import os
import time
import signal

import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware.camera import Camera
from hardware.ir_leds import IRLEDController
from detection.landmarks import FaceLandmarker
from detection.ear import compute_avg_ear
from detection.mar import compute_mar
from detection.head_pose import estimate_head_pose
from detection.perclos import DrowsinessDetector
from detection.blendshapes import eye_closure_score, jaw_open_score
from scoring.state_machine import StateMachine, DrowsinessState
from scoring.fusion import compute_drowsiness_score
from alerts.alert_system import AlertSystem
from calibration.calibrator import Calibrator
from data_logging.data_logger import DataLogger
from config import (
    FACE_LOST_WARNING,
    FACE_LOST_CRITICAL,
    EYE_CLOSURE_SIGNAL,
    YAWN_SIGNAL,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Drive Safe Drowsiness Detection")
    parser.add_argument(
        "--source", default="0",
        help="Camera source: integer for webcam index, file path for video, 'picamera2' for RPi camera",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration")
    parser.add_argument("--no-display", action="store_true", help="Headless mode")
    parser.add_argument("--no-alerts", action="store_true", help="Disable audio alerts")
    return parser.parse_args()


def parse_source(source_str):
    """Convert source string to appropriate type."""
    if source_str == "picamera2":
        return "picamera2"
    try:
        return int(source_str)
    except ValueError:
        return source_str  # file path


STATE_COLORS = {
    DrowsinessState.ALERT: (0, 255, 0),      # green
    DrowsinessState.MILD: (0, 255, 255),      # yellow
    DrowsinessState.MODERATE: (0, 165, 255),  # orange
    DrowsinessState.SEVERE: (0, 0, 255),      # red
}


def draw_overlay(frame, state, perclos, ear, mar, eye_closure, jaw_open,
                 pitch, yaw, score, fps):
    """Draw debug info on frame."""
    color = STATE_COLORS.get(state, (255, 255, 255))
    active_marker = lambda is_active: ">" if is_active else " "
    eye_active = EYE_CLOSURE_SIGNAL == "blendshape"
    yawn_active = YAWN_SIGNAL == "blendshape"

    lines = [
        (f"State: {state.name}", 30, 0.8, color, 2),
        (f"PERCLOS: {perclos:.3f}", 60, 0.6, (255, 255, 255), 1),
        (f"{active_marker(not eye_active)}EAR:        {ear:.3f}", 85, 0.6,
         (255, 255, 255), 1),
        (f"{active_marker(eye_active)}eyeClosure: {eye_closure:.3f}", 110,
         0.6, (255, 255, 255), 1),
        (f"{active_marker(not yawn_active)}MAR:        {mar:.3f}", 135, 0.6,
         (255, 255, 255), 1),
        (f"{active_marker(yawn_active)}jawOpen:    {jaw_open:.3f}", 160, 0.6,
         (255, 255, 255), 1),
        (f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}", 185, 0.6, (255, 255, 255), 1),
        (f"Score: {score:.3f}", 210, 0.6, (255, 255, 255), 1),
        (f"FPS: {fps:.0f}", 235, 0.6, (255, 255, 255), 1),
    ]
    for text, y, scale, c, thick in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thick)

    # State indicator bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 5), color, -1)

    return frame


def main():
    args = parse_args()
    source = parse_source(args.source)

    print("=" * 50)
    print("  Drive Safe 2.0 — Drowsiness Detection")
    print("=" * 50)
    print()
    print("DISCLAIMER: This is a supplementary aid,")
    print("NOT a certified safety device.")
    print()

    # Initialize components
    print(f"[Init] Camera source: {source}")
    camera = Camera(source)

    ir_leds = IRLEDController()
    ir_leds.on()

    print("[Init] Loading MediaPipe face landmarker...")
    landmarker = FaceLandmarker()

    detector = DrowsinessDetector()
    state_machine = StateMachine()
    alert_system = AlertSystem() if not args.no_alerts else None
    logger = DataLogger()

    # Global start time — shared across calibration and main loop
    # so MediaPipe timestamps are always monotonically increasing
    start_time = time.time()

    # Calibration
    if not args.no_calibration:
        calibrator = Calibrator()
        thresholds = calibrator.calibrate(camera, landmarker, start_time)
        if thresholds:
            detector.apply_calibration(thresholds)
            logger.log_event("CALIBRATION_DONE", str(thresholds))
        else:
            print("[Calibration] Using default thresholds.")
    else:
        print("[Calibration] Skipped — using defaults.")

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    print("[Running] Detection active. Press 'q' or Ctrl+C to quit.")
    frame_count = 0
    face_lost_since = None
    fps = 0
    fps_timer = time.time()
    fps_count = 0

    try:
        while running:
            success, frame = camera.read()
            if not success:
                if isinstance(source, str) and source != "picamera2":
                    print("[Done] End of video file.")
                    break
                continue

            frame_count += 1
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            result = landmarker.detect(frame_rgb, timestamp_ms)

            if result is None:
                # Face lost
                now = time.time()
                if face_lost_since is None:
                    face_lost_since = now
                lost_duration = now - face_lost_since

                if lost_duration > FACE_LOST_CRITICAL:
                    print(f"[Warning] Face lost for {lost_duration:.0f}s")
                    logger.log_event("FACE_LOST_EXTENDED")
                elif lost_duration > FACE_LOST_WARNING:
                    print("[Info] Please adjust position so camera can see your face.")

                if not args.no_display:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Drive Safe", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            face_lost_since = None

            # Compute metrics — both EAR/MAR (geometric) and blendshape (learned)
            # signals are computed every frame so the overlay can show them side
            # by side. The active signal feeds the detector.
            left_eye, right_eye = landmarker.get_eye_landmarks(result, frame.shape)
            ear = compute_avg_ear(left_eye, right_eye)

            mouth = landmarker.get_mouth_landmarks(result, frame.shape)
            mar = compute_mar(mouth)

            blendshapes = landmarker.get_blendshapes(result)
            eye_closure = eye_closure_score(blendshapes)
            jaw_open = jaw_open_score(blendshapes)

            if EYE_CLOSURE_SIGNAL == "blendshape" and blendshapes is not None:
                perclos = detector.update_eye_closure_score(eye_closure)
            else:
                perclos = detector.update_ear(ear)

            if YAWN_SIGNAL == "blendshape" and blendshapes is not None:
                detector.update_jaw_open_score(jaw_open)
            else:
                detector.update_mar(mar)

            head_points = landmarker.get_head_pose_landmarks(result, frame.shape)
            pitch, yaw, roll = estimate_head_pose(head_points, frame.shape)
            head_nod = detector.check_head_nod(pitch)
            looking_away = detector.check_looking_away(yaw)

            # Update state machine
            state = state_machine.update(
                perclos=perclos,
                avg_blink_dur=detector.get_avg_blink_duration(),
                yawn_count_5min=detector.yawn_count_5min,
                head_nod=head_nod,
                eyes_closed_duration=detector.current_closure_duration(),
            )

            # Compute composite score
            score = compute_drowsiness_score(
                perclos, detector.get_avg_blink_duration(),
                detector.yawn_count_5min, head_nod, looking_away,
            )

            # Alerts
            if alert_system:
                if state < DrowsinessState.SEVERE:
                    alert_system.stop_alarm()
                if state != DrowsinessState.ALERT:
                    alert_system.alert(state)

            # Log
            logger.log(perclos, ear, mar, pitch, yaw, roll, state, score)

            # FPS calculation
            fps_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

            # Display
            if not args.no_display:
                frame = draw_overlay(frame, state, perclos, ear, mar,
                                     eye_closure, jaw_open, pitch, yaw,
                                     score, fps)
                cv2.imshow("Drive Safe", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_count % 15 == 0:
                print(f"[{state.name:<8}] EAR={ear:.3f} eyeClose={eye_closure:.3f} "
                      f"PERCLOS={perclos:.3f} MAR={mar:.3f} jawOpen={jaw_open:.3f} "
                      f"pitch={pitch:+.1f} score={score:.2f} fps={fps:.1f}")

    finally:
        print("\n[Shutdown] Cleaning up...")
        ir_leds.cleanup()
        if alert_system:
            alert_system.stop_alarm()
            alert_system.cleanup()
        logger.close()
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[Shutdown] Done.")


if __name__ == "__main__":
    main()
