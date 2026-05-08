# Drive Safe — Raspberry Pi setup

Validated on **Raspberry Pi 5 (4GB)** running **Pi OS Trixie** (Debian 13, system Python 3.13), Camera Module v3 with autofocus, official active cooler. ~30 fps end-to-end.

## 1. System packages

```bash
sudo apt update
sudo apt install -y libcamera-tools libcap-dev v4l-utils
```

`libcamera-tools` provides `libcamerify` (the V4L2→libcamera shim we depend on). `libcap-dev` is needed because `picamera2` pulls in `python-prctl`, which has no wheel and compiles from source.

## 2. Python environment

System Python on Trixie is 3.13, but MediaPipe doesn't ship aarch64 wheels for 3.13 yet. Install Python 3.11 in user space via `uv` and create a venv against it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL              # reload PATH

uv python install 3.11
uv venv ~/venvs/cv --python 3.11
source ~/venvs/cv/bin/activate

uv pip install -r rpi/requirements-rpi.txt
```

Note: `picamera2` is installed but doesn't actually run in this venv — its `libcamera` Python binding is built against system Python 3.13 and is ABI-incompatible with 3.11. We use the libcamerify path instead (see below).

## 3. MediaPipe model

The face landmarker `.task` blob is gitignored. Fetch it once:

```bash
mkdir -p rpi/models
wget -O rpi/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 4. Run

The MediaPipe model path is resolved relative to CWD, so **run from inside `rpi/`**. Wrap the command in `libcamerify` so OpenCV's V4L2 capture is proxied to the Pi camera through libcamera:

```bash
cd rpi
libcamerify python main.py --source 0 --no-alerts
```

Useful flags:
- `--no-display` — skip the OpenCV window; print a state line per second to stdout instead.
- `--no-calibration` — skip the 10-second baseline capture.
- `--no-alerts` — disable audio alerts (Pi 5 has no built-in audio out; needs USB/HDMI/BT speaker).

## Known issues / gotchas

- **OpenCV 4.13 breaks the libcamerify path.** Stay on `opencv-contrib-python==4.11.0.86`. Do not let pip resolve it forward.
- **OpenCV must be `opencv-contrib-python`, not `opencv-python` or `*-headless`.** The contrib aarch64 wheel ships with QT5 GUI; the others on aarch64 are headless. Installing multiple variants causes the headless one to win the import-order race — uninstall all and install only `opencv-contrib-python`.
- **First several `cv2.VideoCapture` reads return `False`** under libcamerify. `Camera._init_opencv` warms up before returning.
- **Frames arrive as `(1, H*W*3)` flat RGB888 buffers**, not `(H, W, 3) BGR`. `Camera.read` reshapes and converts.
- **`RPi.GPIO` doesn't work on Pi 5** — the RP1 chip moved GPIO off the SoC. The `IRLEDController` prints a non-fatal warning and continues. Replace with `rpi-lgpio` if/when IR LEDs are actually wired.
- **Autofocus isn't accessible** through the libcamerify path (only via `picamera2`). The Camera Module v3 stays at libcamera's default focus. To unlock AF + exposure controls, build libcamera's Python bindings against the venv's Python 3.11 — deferred work.
