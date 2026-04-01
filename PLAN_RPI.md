# Drive Safe 2.0 — Raspberry Pi Detailed Plan

## Hardware

### Recommended Configuration: RPi 5 4GB + Hailo-8L

| Component | Specific Product | Price (USD) | Notes |
|-----------|-----------------|-------------|-------|
| Board | Raspberry Pi 5 4GB | $60 | Sweet spot — 2GB works but no headroom |
| AI Accelerator | RPi AI Kit (Hailo-8L + M.2 HAT+) | $70 | 13 TOPS, official RPi support. Optional but strongly recommended |
| Camera | RPi Camera Module 3 NoIR | $35 | No IR filter = sensitive to IR light for night driving |
| IR LEDs | 850nm IR LED board (x2) | $5-8 | Mount flanking camera. Invisible to human eye at 940nm, faint red glow at 850nm |
| Speaker | Piezo buzzer + small 3W powered speaker | $3-5 | Buzzer for alarm, speaker for TTS voice alerts |
| Power Supply | Car 12V→5V USB-C buck converter (5A rated) | $12-18 | Must be automotive-grade with input protection |
| Storage | 32GB A2-rated MicroSD (Samsung/SanDisk) | $8-10 | A2 speed class for faster random I/O |
| Cooling | Heatsink + small fan | $5 | Essential for continuous operation in car heat |
| Case | 3D-printed custom enclosure | $5-10 | Must accommodate camera, IR LEDs, HAT+. Or ~$15-20 bought |
| Mounting | Suction cup mount + adjustable arm | $8-12 | Dashboard or windshield mount |
| Cables | CSI ribbon cable, jumper wires, USB-C cable | $5-8 | |
| Optional | 1.3" OLED (SSD1306) or 2" LCD | $5-12 | Status display: state, FPS, score |

### Cost Summary

| Configuration | Total |
|---------------|-------|
| Minimum viable (no accelerator, no display) | $145-165 |
| Recommended (with Hailo-8L) | $215-235 |
| Full featured (accelerator + display) | $225-250 |

### Power Supply Details

Car electrical systems are noisy (12-14.4V nominal, spikes up to 40V+ during load dump). Requirements:

- Quality buck converter with input protection (TVS diode, reverse polarity protection)
- Output: 5V 5A minimum — RPi 5 draws 3-4A under load with peripherals
- Consider automotive-grade converter (e.g., Pololu 5V 5A step-down) or a quality car USB-C PD adapter rated for RPi 5
- Optional: supercapacitor or graceful shutdown circuit to handle engine cranking voltage drops
- Wire directly to a switched 12V source (on with ignition) so the system starts/stops with the car

### Camera Options

| Camera | Price | Night Vision | Resolution | Notes |
|--------|-------|-------------|------------|-------|
| RPi Camera Module 3 NoIR | $35 | Yes (with IR LEDs) | 12MP, 1080p video | Best choice |
| Arducam OV5647 NoIR | $12-15 | Yes (with IR LEDs) | 5MP | Budget option, works fine |
| USB camera with IR (e.g., ELP) | $25-40 | Built-in IR LEDs | 2MP | Convenient but adds USB latency |

Recommendation: RPi Camera Module 3 NoIR. Native MIPI CSI interface = lowest latency and best integration with picamera2.

### Mounting

- Camera should be 30-50cm from driver's face, pointing at the driver
- Must NOT obstruct driver's forward view
- Options: suction cup on windshield edge, clip on rearview mirror stalk, adhesive bracket on dashboard
- IR LEDs mounted flanking the camera, angled toward the driver's face
- Ensure the ribbon cable is secured (vibration can disconnect it — use tape or a clip)

---

## Software Stack

| Layer | Technology |
|-------|-----------|
| OS | Raspberry Pi OS Lite 64-bit (headless) |
| Python | 3.11+ |
| Face detection/landmarks | MediaPipe (primary) or PFLD via TFLite (fallback) |
| Camera capture | picamera2 |
| Computer vision | OpenCV |
| Audio playback | pygame.mixer |
| Text-to-speech | pyttsx3 or espeak |
| AI accelerator runtime | Hailo TAPPAS + rpicam-apps (if using Hailo-8L) |
| Data logging | SQLite3 |
| Process management | systemd service + watchdog |
| Optional display | luma.oled (for SSD1306 OLED) |

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Face landmark FPS (CPU-only) | 15-25 FPS | MediaPipe on RPi 5, 640x480 |
| Face landmark FPS (Hailo-8L) | 30-60 FPS | With AI accelerator |
| End-to-end pipeline latency | <100ms/frame | Capture → inference → scoring |
| Detection → alert latency | <500ms | Eye closure to alarm sound |
| Night detection accuracy | Within 5% of daytime | IR camera compensates |
| Continuous operation | 8+ hours | No thermal throttling |
| Boot to ready | <60 seconds | Including systemd startup and calibration |

---

## Development Plan (Weeks 5-8)

### Week 5: Hardware Assembly & Camera Pipeline

#### Task 5.1: Hardware Assembly

```
1. Flash RPi OS Lite 64-bit to MicroSD
2. Attach heatsink + fan to RPi 5
3. If using Hailo: attach M.2 HAT+ board, insert Hailo-8L module
4. Connect NoIR camera via CSI ribbon cable
5. Wire IR LED board (can be powered from RPi 5V GPIO pins via a transistor for on/off control)
6. Connect speaker to 3.5mm audio jack or USB
7. Boot and verify: camera, audio, GPIO, Hailo (if present)
```

#### Task 5.2: Camera Pipeline Setup

```python
# Core capture loop with picamera2
from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    # process frame...
```

- Capture at 640x480 RGB at 30 FPS
- Process every frame if accelerator is present, or every 2nd frame if CPU-only
- Verify IR illumination produces clear face images in complete darkness
- Test and adjust exposure, gain, and white balance for IR-only illumination
- Test varying ambient light: streetlights, oncoming headlights, dashboard glow

#### Task 5.3: IR LED Control

```python
# GPIO control for IR LEDs
import RPi.GPIO as GPIO

IR_LED_PIN = 17  # BCM pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_LED_PIN, GPIO.OUT)

def ir_leds_on():
    GPIO.output(IR_LED_PIN, GPIO.HIGH)

def ir_leds_off():
    GPIO.output(IR_LED_PIN, GPIO.LOW)
```

- IR LEDs should be always-on during operation (they're invisible to the driver)
- Use a MOSFET or transistor to switch LED power from GPIO — don't draw high current directly from GPIO pins
- Test both 850nm (faint red glow visible) and 940nm (invisible but lower camera sensitivity) if you have both

### Week 6: Detection Algorithm Port

#### Task 6.1: MediaPipe Face Landmarker Setup

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize face landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,  # critical for drowsiness signals
)
landmarker = vision.FaceLandmarker.create_from_options(options)
```

If MediaPipe blendshapes are available, use them directly:
- `eyeBlinkLeft` / `eyeBlinkRight` (0.0=open, 1.0=closed) — use for PERCLOS
- `jawOpen` — use for yawn detection
- Head pose from 3D landmark geometry

If using raw landmarks, compute EAR and MAR manually (see algorithm spec in PROJECT_PLAN.md).

#### Task 6.2: EAR and PERCLOS Implementation

```python
import numpy as np
from collections import deque
import time

class DrowsinessDetector:
    def __init__(self, fps=15, window_seconds=60):
        self.ear_threshold = 0.20  # will be calibrated
        self.perclos_window = deque(maxlen=fps * window_seconds)
        self.blink_durations = deque(maxlen=100)
        self.current_blink_start = None

    def compute_ear(self, eye_landmarks):
        """Compute Eye Aspect Ratio from 6 landmarks per eye."""
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def update_perclos(self, ear):
        """Update PERCLOS with new EAR reading."""
        is_closed = ear < self.ear_threshold
        self.perclos_window.append(1 if is_closed else 0)

        # Track blink duration
        if is_closed and self.current_blink_start is None:
            self.current_blink_start = time.time()
        elif not is_closed and self.current_blink_start is not None:
            duration = time.time() - self.current_blink_start
            self.blink_durations.append(duration)
            self.current_blink_start = None

        return sum(self.perclos_window) / len(self.perclos_window) if self.perclos_window else 0

    def get_avg_blink_duration(self, last_n=10):
        """Average blink duration over last N blinks."""
        recent = list(self.blink_durations)[-last_n:]
        return sum(recent) / len(recent) if recent else 0
```

#### Task 6.3: Yawn Detection

```python
def compute_mar(self, mouth_landmarks):
    """Compute Mouth Aspect Ratio from mouth landmarks."""
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_landmarks
    vertical_1 = np.linalg.norm(p2 - p8)
    vertical_2 = np.linalg.norm(p3 - p7)
    vertical_3 = np.linalg.norm(p4 - p6)
    horizontal = np.linalg.norm(p1 - p5)
    return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
```

- Yawn = MAR > 0.6 sustained for >2 seconds
- Track yawn count in rolling 5-minute window
- 3+ yawns = drowsiness signal

#### Task 6.4: Head Pose Estimation

```python
import cv2

# 3D model points (generic face model)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye corner
    (225.0, 170.0, -135.0), # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0) # Right mouth corner
])

def estimate_head_pose(self, image_points, frame_shape):
    """Estimate head pitch, yaw, roll from facial landmarks."""
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll
```

- Head nod: pitch drops >15 degrees below calibrated neutral
- Looking away: |yaw| > 30 degrees for >2 seconds
- Head tilt: |roll| > 20 degrees sustained

#### Task 6.5: State Machine Implementation

```python
from enum import Enum

class DrowsinessState(Enum):
    ALERT = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3

class StateMachine:
    def __init__(self):
        self.state = DrowsinessState.ALERT
        self.state_entry_time = time.time()
        self.moderate_count_15min = 0

    def update(self, perclos, avg_blink_dur, yawn_count_5min, head_nod, eyes_closed_duration):
        previous = self.state

        if self.state == DrowsinessState.ALERT:
            if perclos > 0.15 or yawn_count_5min > 3:
                self.transition(DrowsinessState.MILD)

        elif self.state == DrowsinessState.MILD:
            active_signals = sum([
                perclos > 0.15,
                avg_blink_dur > 0.5,
                yawn_count_5min > 3,
                head_nod
            ])
            if active_signals >= 2 or perclos > 0.20:
                self.transition(DrowsinessState.MODERATE)
            elif perclos < 0.10 and not head_nod and yawn_count_5min < 2:
                if time.time() - self.state_entry_time > 30:
                    self.transition(DrowsinessState.ALERT)

        elif self.state == DrowsinessState.MODERATE:
            if perclos > 0.30 or head_nod or eyes_closed_duration > 2.0:
                self.transition(DrowsinessState.SEVERE)
            elif perclos < 0.10 and not head_nod:
                if time.time() - self.state_entry_time > 30:
                    self.transition(DrowsinessState.ALERT)

        elif self.state == DrowsinessState.SEVERE:
            if perclos < 0.10 and not head_nod and eyes_closed_duration < 0.5:
                if time.time() - self.state_entry_time > 30:
                    self.transition(DrowsinessState.MODERATE)

        return self.state

    def transition(self, new_state):
        if new_state == DrowsinessState.MODERATE:
            self.moderate_count_15min += 1
        self.state = new_state
        self.state_entry_time = time.time()
```

### Week 7: Night Driving, Alerts & Calibration

#### Task 7.1: Night Driving Validation

Test checklist:
- [ ] Complete darkness with IR LEDs only — verify clear face image
- [ ] Landmark detection accuracy under IR-only illumination
- [ ] Varying ambient light (streetlights, oncoming headlights)
- [ ] Drivers wearing clear glasses (IR reflection off lenses is a known issue)
- [ ] Drivers wearing prescription glasses with anti-reflective coating
- [ ] Transition: bright → dark and dark → bright (tunnel entry/exit)
- [ ] Dashboard illumination only (instrument cluster glow)

If glasses cause IR reflection issues:
- Adjust IR LED angle (avoid perpendicular to lens surface)
- Try 940nm instead of 850nm
- Detect glasses via landmarks and fall back to yawn + head pose only

#### Task 7.2: Audio Alert System

```python
import pygame
import pyttsx3

class AlertSystem:
    def __init__(self):
        pygame.mixer.init()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)

        self.sounds = {
            DrowsinessState.MILD: pygame.mixer.Sound('chime_soft.wav'),
            DrowsinessState.MODERATE: pygame.mixer.Sound('alarm_medium.wav'),
            DrowsinessState.SEVERE: pygame.mixer.Sound('alarm_loud.wav'),
        }
        self.last_alert_time = {}
        self.cooldown = {
            DrowsinessState.MILD: 120,     # 2 min cooldown
            DrowsinessState.MODERATE: 60,   # 1 min cooldown
            DrowsinessState.SEVERE: 0,      # no cooldown — always alert
        }

    def alert(self, state):
        now = time.time()
        last = self.last_alert_time.get(state, 0)

        if now - last < self.cooldown.get(state, 0):
            return  # in cooldown

        if state == DrowsinessState.MILD:
            self.sounds[state].play()

        elif state == DrowsinessState.MODERATE:
            self.sounds[state].play()
            self.tts.say("You appear drowsy. Consider taking a break.")
            self.tts.runAndWait()

        elif state == DrowsinessState.SEVERE:
            self.sounds[state].play(-1)  # loop continuously
            self.tts.say("Warning! Pull over safely now!")
            self.tts.runAndWait()

        self.last_alert_time[state] = now
```

Use a powered speaker (not just a piezo buzzer) for voice alerts. Connect via 3.5mm audio jack or USB audio adapter.

#### Task 7.3: Calibration Routine

```python
class Calibrator:
    def __init__(self, duration=10, fps=15):
        self.duration = duration
        self.fps = fps

    def calibrate(self, detector, landmarker, picam2):
        """Run 10-second calibration. Returns thresholds."""
        print("Calibration: Look at the camera naturally for 10 seconds...")
        # Play TTS instruction

        ear_samples = []
        mar_samples = []
        pitch_samples = []

        start = time.time()
        while time.time() - start < self.duration:
            frame = picam2.capture_array()
            result = landmarker.detect(frame)

            if result.face_landmarks:
                ear = detector.compute_ear(...)  # extract eye landmarks
                mar = detector.compute_mar(...)  # extract mouth landmarks
                pitch, _, _ = detector.estimate_head_pose(...)

                ear_samples.append(ear)
                mar_samples.append(mar)
                pitch_samples.append(pitch)

        ear_mean = np.mean(ear_samples)
        ear_std = np.std(ear_samples)

        thresholds = {
            'ear_closed': max(ear_mean - 2 * ear_std, ear_mean * 0.6),
            'mar_yawn': np.mean(mar_samples) + 3 * np.std(mar_samples),
            'pitch_neutral': np.mean(pitch_samples),
            'pitch_nod_threshold': np.mean(pitch_samples) - 15,
        }

        print(f"Calibration complete. EAR baseline: {ear_mean:.3f}, threshold: {thresholds['ear_closed']:.3f}")
        return thresholds
```

### Week 8: System Integration & Robustness

#### Task 8.1: Main Loop

```python
def main():
    # Initialize hardware
    picam2 = init_camera()
    ir_leds_on()

    # Initialize detection
    landmarker = init_mediapipe()
    detector = DrowsinessDetector()
    state_machine = StateMachine()
    alert_system = AlertSystem()
    logger = DataLogger('drowsiness_log.db')

    # Calibration
    calibrator = Calibrator()
    thresholds = calibrator.calibrate(detector, landmarker, picam2)
    detector.ear_threshold = thresholds['ear_closed']

    # Main detection loop
    while True:
        frame = picam2.capture_array()
        result = landmarker.detect(frame)

        if not result.face_landmarks:
            handle_face_lost()
            continue

        # Compute metrics
        ear = detector.compute_ear(extract_eye_landmarks(result))
        perclos = detector.update_perclos(ear)
        mar = detector.compute_mar(extract_mouth_landmarks(result))
        yawn_detected = detector.check_yawn(mar)
        pitch, yaw, roll = detector.estimate_head_pose(...)
        head_nod = detector.check_head_nod(pitch, thresholds['pitch_nod_threshold'])

        # Update state machine
        state = state_machine.update(
            perclos=perclos,
            avg_blink_dur=detector.get_avg_blink_duration(),
            yawn_count_5min=detector.yawn_count_5min,
            head_nod=head_nod,
            eyes_closed_duration=detector.current_closure_duration()
        )

        # Trigger alerts
        if state != DrowsinessState.ALERT:
            alert_system.alert(state)

        # Log data
        logger.log(perclos, ear, mar, pitch, yaw, roll, state)
```

#### Task 8.2: Face Loss Recovery

```python
def handle_face_lost():
    """Handle case where face is not detected."""
    if face_lost_duration > 5:
        alert_system.play_attention_sound()
        # "Please adjust your position so the camera can see your face"
    if face_lost_duration > 30:
        alert_system.play_system_warning()
        logger.log_event("FACE_LOST_EXTENDED")
    # Continue monitoring — face may reappear
```

#### Task 8.3: Data Logging

```python
import sqlite3

class DataLogger:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS readings (
            timestamp REAL, perclos REAL, ear REAL, mar REAL,
            pitch REAL, yaw REAL, roll REAL, state TEXT
        )''')

    def log(self, perclos, ear, mar, pitch, yaw, roll, state):
        self.conn.execute(
            'INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (time.time(), perclos, ear, mar, pitch, yaw, roll, state.name)
        )
        self.conn.commit()  # batch commits for performance in production
```

Log raw metrics for:
- Post-drive analysis
- Threshold tuning
- Potential future custom ML model training

Do NOT log raw images — privacy concern and storage bloat.

#### Task 8.4: Auto-Start and Watchdog

Create systemd service: `/etc/systemd/system/drivesafe.service`
```ini
[Unit]
Description=Drive Safe Drowsiness Detection
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/drivesafe
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=5
WatchdogSec=30

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable drivesafe
sudo systemctl start drivesafe
```

The `WatchdogSec=30` means systemd will restart the service if it doesn't send a heartbeat within 30 seconds. Use `systemd.daemon.notify("WATCHDOG=1")` in the main loop.

#### Task 8.5: Graceful Shutdown

- Wire a physical button to a GPIO pin
- On press: clean shutdown sequence (stop detection, save logs, `sudo shutdown -h now`)
- On power loss (car turned off): systemd handles cleanup; SQLite journal ensures no data corruption

---

## Testing Checklist (RPi-Specific)

### Hardware Validation
- [ ] Camera produces clear image at 640x480 in daylight
- [ ] IR LEDs illuminate face clearly in complete darkness
- [ ] No visible IR glow to driver (940nm) or minimal glow (850nm)
- [ ] Speaker volume is audible over car noise (road noise, A/C, radio)
- [ ] System boots and starts detection within 60 seconds
- [ ] System runs continuously for 4+ hours without crash or performance degradation
- [ ] Power supply handles engine start voltage drop without rebooting Pi
- [ ] Camera ribbon cable stays connected under car vibration (test on rough road)
- [ ] System operates in car cabin temperatures (0-50 C)

### Detection Validation
- [ ] PERCLOS correctly identifies simulated drowsiness (slow eye closure)
- [ ] Normal blinking does NOT trigger alerts
- [ ] Yawning is detected accurately (MAR threshold)
- [ ] Head nodding triggers appropriate alert level
- [ ] Sunglasses: system detects them and falls back to yawn + head pose
- [ ] Glasses with IR: test reflection, adjust LED angle if needed
- [ ] Multiple lighting transitions handled (tunnel, night/day)
- [ ] System recovers when face re-enters frame after being lost

---

## Directory Structure

```
/home/pi/drivesafe/
├── main.py                 # Entry point and main loop
├── detection/
│   ├── landmarks.py        # MediaPipe face landmarker wrapper
│   ├── ear.py              # Eye Aspect Ratio computation
│   ├── mar.py              # Mouth Aspect Ratio computation
│   ├── head_pose.py        # Head pose estimation (solvePnP)
│   └── perclos.py          # PERCLOS rolling window calculator
├── scoring/
│   ├── state_machine.py    # Drowsiness state machine
│   └── fusion.py           # Multi-signal weighted scoring
├── alerts/
│   ├── alert_system.py     # Sound and TTS alert management
│   └── sounds/             # Audio files (chime, alarm, etc.)
├── calibration/
│   └── calibrator.py       # Per-driver calibration routine
├── hardware/
│   ├── camera.py           # picamera2 wrapper
│   └── ir_leds.py          # GPIO IR LED control
├── logging/
│   └── data_logger.py      # SQLite data logging
├── config.py               # Thresholds, weights, settings
├── models/
│   └── face_landmarker.task # MediaPipe model file
└── requirements.txt
```

## Dependencies (requirements.txt)

```
mediapipe>=0.10.9
opencv-python-headless>=4.8
numpy>=1.24
picamera2>=0.3
pygame>=2.5
pyttsx3>=2.90
RPi.GPIO>=0.7
```
