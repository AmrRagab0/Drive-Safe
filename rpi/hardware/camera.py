import cv2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, IS_RPI


class Camera:
    """Unified camera interface for PC (webcam/video) and RPi (picamera2)."""

    def __init__(self, source=0):
        """
        Args:
            source: int for webcam index (0 = default/DroidCam),
                    str for video file path,
                    "picamera2" for RPi native camera.
        """
        self.source = source
        self.cap = None
        self.picam2 = None

        if source == "picamera2":
            if not IS_RPI:
                raise RuntimeError("picamera2 is only available on Raspberry Pi")
            self._init_picamera2()
        else:
            self._init_opencv(source)

    def _init_opencv(self, source):
        import time
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # libcamerify warmup: first several reads return False before frames flow.
        time.sleep(0.5)
        for _ in range(15):
            ok, _ = self.cap.read()
            if ok:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Camera opened but never delivered a frame")

    def _init_picamera2(self):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        """Returns (success, frame) tuple. Frame is BGR for OpenCV, RGB for picamera2."""
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            return True, frame
        ret, frame = self.cap.read()
        # libcamerify delivers RGB888 as a flat (1, H*W*3) buffer — reshape + swap to BGR.
        if (ret and frame is not None and frame.ndim == 2
                and frame.size == CAMERA_WIDTH * CAMERA_HEIGHT * 3):
            frame = frame.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return ret, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
