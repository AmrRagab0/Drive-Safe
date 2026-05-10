# Standalone Pi-class Device — Phased Plan

## Context

The Flutter phone app (`origin/main`) using Google ML Kit detects drowsiness with high accuracy. The goal is now to ship the same capability as a **standalone in-car device** (Pi-class hardware), not on a phone. Night/IR support is nice-to-have, not required for v1.

Direct "deploy the Flutter app on the Pi" is a dead end:
- Google ML Kit ships only mobile binaries — **no Linux/ARM build** of the underlying detector exists. The plugin `google_mlkit_face_detection` cannot link on the Pi.
- The accuracy the user values comes mostly from ML Kit's pre-trained **`eyeOpenProbability` classifier head**, not from BlazeFace itself.
- **MediaPipe FaceLandmarker** — already used in the Pi prototype — exposes `eyeBlinkLeft` / `eyeBlinkRight` / `jawOpen` **blendshapes** that are direct equivalents of ML Kit's probabilities. Crucially, `landmarks.py` already sets `output_face_blendshapes=True`, but `main.py` discards the signal. **Wiring this up is most of the accuracy gap.**

So the real plan is: **match the phone app's accuracy on Pi by activating signals MediaPipe is already computing**, then optionally reuse the Flutter UI via flutter-pi for visual parity.

## Decision

Build a standalone Pi 5 device with:
1. **Detection brain**: extend the existing Python/MediaPipe prototype with blendshape-based eye-closure and yawn signals (the ML Kit `eyeOpenProbability` / `jawOpen` equivalents). Train a TFLite eye-state head only if blendshapes prove insufficient.
2. **UI**: reuse the Flutter UI from the phone app via `flutter-pi`, talking to the Python brain over a local Unix socket. Falls back to the existing OpenCV preview if the Flutter port slips.
3. **Chassis**: Pi 5 + Camera Module v3 + 7" touchscreen + USB/BT speaker + 12V→USB-C power. (Pi 5 + Camera Module v3 already in hand.)

This decoupling lets us ship detection improvements without touching UI, keep the ML stack portable (MediaPipe + blendshapes also run on phone — future unification with the Flutter app's detector is feasible), and sidestep flutter-pi's plugin limitations (no ML plugin needed at all on the Flutter side).

## Phases

### Phase 0 — Verify current Pi prototype runs *(DONE)*

User confirmed the prototype runs on Pi 5 / Trixie via `libcamerify python main.py --source 0 --no-alerts` from `rpi/`. ~30fps end-to-end. Calibration completes; state line prints in `--no-display` mode.

### Phase 1 — Close the accuracy gap (current focus)

**Sub-phase 1.1 — Blendshape-based eye-closure signal.**
MediaPipe already computes `eyeBlinkLeft` / `eyeBlinkRight` (0.0 = open, 1.0 = fully closed). These are the learned equivalent of ML Kit's `eyeOpenProbability`. EAR is a hand-rolled geometric ratio and noisier.
- Read blendshapes in `main.py` and feed them into PERCLOS as the primary closure signal.
- Keep EAR computation as a side-by-side overlay for comparison; choose default via config flag.
- Threshold: start at `max(eyeBlinkLeft, eyeBlinkRight) > 0.5` (matches Flutter app's `< 0.6 eyeOpenProbability` after inversion). Will be calibration-overridden.

**Sub-phase 1.2 — Blendshape-based yawn signal.**
Use `jawOpen` blendshape as primary yawn signal alongside MAR. Same calibration pattern.

**Sub-phase 1.3 — Calibrate blendshape thresholds + update overlay.**
Extend `Calibrator` to capture baseline eye-blink and jaw-open values during the 10s baseline window. Show both new signals in the OpenCV overlay so we can validate behavior visually before deciding which signal to keep.

**Sub-phase 1.4 — Validation against the phone app.**
Record matched clips (same driver, same lighting) on both phone and Pi. Compare alarm triggers. Target ≤10% disagreement before declaring Phase 1 complete. If blendshape signal underperforms ML Kit, *then* train a small TFLite eye-state classifier (deferred follow-up).

### Phase 2 — UI parity via flutter-pi

- Install `flutter-pi` on Pi 5 (DRM/KMS rendering, no X11). Repo: `ardera/flutter-pi`.
- Pull the full Flutter source project (user has it locally — not yet on this machine; only loose `.dart` files exist on `origin/main`).
- Strip `google_mlkit_face_detection` and `camera` plugins. Replace `painter.dart`'s detection callback with reads from a **Unix domain socket** publishing `{state, eyeClosure, perclos, headPitch, score}` JSON from the Python brain at ~10 Hz.
- Keep `audioplayers` for the alarm (has a Linux backend that works under flutter-pi).
- Build for ARM64 with `flutter build bundle` + flutter-pi runner.

### Phase 3 — Device chassis

- 7" Pi touchscreen, USB or BT speaker, 12V cigarette → USB-C PD power supply, 3D-printed dash mount.
- Systemd unit to autostart the Python brain + flutter-pi runner on boot.
- Watchdog: restart on crash, log to `rpi/drowsiness_log.db`.

### Phase 4 (deferred) — Night driving

- NoIR camera + IR LED ring (the `IRLEDController` stub already exists in `rpi/hardware/`).
- Replace `RPi.GPIO` (broken on Pi 5) with `rpi-lgpio`.
- Retrain eye-state classifier (if introduced in Phase 1.4) on IR imagery (RT-BENE has IR splits).

## Critical files

| Path | Role |
|---|---|
| `rpi/main.py` | Pipeline entrypoint — will start consuming blendshapes in Phase 1.1. |
| `rpi/detection/landmarks.py` | Already calls `output_face_blendshapes=True`; `get_blendshapes()` exists. |
| `rpi/detection/perclos.py` | `DrowsinessDetector.update_ear` will gain a blendshape sibling. |
| `rpi/detection/ear.py`, `mar.py` | Kept as fallback signals for comparison overlay. |
| `rpi/calibration/calibrator.py` | Extend to baseline blendshape values. |
| `rpi/scoring/fusion.py` | Untouched in Phase 1 (same composite score, better inputs). |
| `rpi/config.py` | Add `EYE_CLOSURE_SIGNAL` and threshold constants. |

## What this plan deliberately is NOT

- Not porting `google_mlkit_face_detection` to Linux (impossible — closed binary).
- Not running the APK via Waydroid (heavy, brittle, camera passthrough is hard, doesn't generalize).
- Not training a TFLite eye-state classifier *yet* — blendshapes are free and already in the pipeline. Train only if Phase 1.4 validation shows blendshapes are insufficient.
- Not committing to flutter-pi until Phase 1 lands — if accuracy is there, a simpler UI (PyQt, web kiosk, or the existing OpenCV preview) could be enough. Flutter parity is aesthetic, not functional.

## Verification per phase

- **Phase 1.1/1.2**: Overlay shows blendshape values alongside EAR/MAR; values look sensible across normal/closed-eye/yawn frames.
- **Phase 1.3**: Calibrated thresholds differ from defaults by a reasonable margin; same driver, same calibration → similar thresholds across runs.
- **Phase 1.4**: Side-by-side recording shows ≤10% alarm-trigger disagreement vs. phone app on the same clip.
- **Phase 2**: Flutter UI renders on Pi screen, reflects state from Python brain, alarm fires via `audioplayers`.
- **Phase 3**: Cold boot → detecting within 30s with no keyboard/mouse.
