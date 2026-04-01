import platform
import sys

# --- Platform Detection ---
IS_RPI = platform.machine().startswith("aarch64") and "linux" in sys.platform

# --- Camera Settings ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# --- Detection Thresholds (defaults, overridden by calibration) ---
EAR_CLOSED_THRESHOLD = 0.20
MAR_YAWN_THRESHOLD = 0.6
YAWN_MIN_DURATION = 2.0  # seconds
YAWN_WINDOW = 300  # 5 minutes in seconds
YAWN_COUNT_THRESHOLD = 3

# --- Head Pose Thresholds ---
HEAD_NOD_PITCH_DROP = 15  # degrees below neutral
HEAD_YAW_THRESHOLD = 30  # degrees off-center
HEAD_ROLL_THRESHOLD = 20  # degrees tilt
HEAD_POSE_DURATION = 2.0  # seconds sustained

# --- PERCLOS ---
PERCLOS_WINDOW_SECONDS = 60
PERCLOS_MILD_THRESHOLD = 0.15
PERCLOS_MODERATE_THRESHOLD = 0.20
PERCLOS_SEVERE_THRESHOLD = 0.30

# --- Blink Duration ---
BLINK_NORMAL_MAX = 0.4  # seconds
BLINK_DROWSY_THRESHOLD = 0.5  # seconds

# --- Scoring Weights ---
WEIGHT_PERCLOS = 0.35
WEIGHT_BLINK = 0.25
WEIGHT_YAWN = 0.15
WEIGHT_HEAD_POSE = 0.15
WEIGHT_GAZE = 0.10

# --- State Machine ---
RECOVERY_TIME = 1  # seconds in improved state before downgrade (30 for production)

# --- Alert Cooldowns (seconds) ---
ALERT_COOLDOWN_MILD = 120
ALERT_COOLDOWN_MODERATE = 60
ALERT_COOLDOWN_SEVERE = 0

# --- Calibration ---
CALIBRATION_DURATION = 10  # seconds

# --- Face Loss ---
FACE_LOST_WARNING = 5  # seconds before "adjust position" prompt
FACE_LOST_CRITICAL = 30  # seconds before system warning

# --- MediaPipe ---
MEDIAPIPE_MODEL_PATH = "models/face_landmarker.task"
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- IR LEDs (RPi only) ---
IR_LED_PIN = 17  # BCM pin
