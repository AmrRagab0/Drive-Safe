# Drive Safe 2.0 — Project Plan

## Overview

| Attribute | Value |
|-----------|-------|
| Duration | 12 weeks |
| Platforms | Android (Flutter) + Raspberry Pi (Python) |
| Team Size | 1-2 developers |
| Budget | ~$225 (RPi recommended config) |
| Goal | Detect driver drowsiness via multi-signal computer vision and alert progressively |

---

## Critique of the 2023 Approach

### What Worked

- Flutter for rapid prototyping — correct choice for a proof of concept
- Google ML Kit is free, on-device, no cloud dependency or API costs
- Core concept validated: a phone camera CAN detect faces and trigger alarms
- Low barrier to entry — no custom model training required

### What Was Critically Wrong

| Problem | Impact | Fix in 2.0 |
|---------|--------|------------|
| Binary eye open/closed detection | ~50%+ false positive rate (alarms on every blink) | PERCLOS over 60-second rolling window |
| Single metric (eye closure only) | Misses drowsy drivers with partially open eyes, yawning, head nodding | Multi-signal fusion: eye + yawn + head pose + gaze |
| No night driving support | System useless during the most dangerous driving hours | IR camera on RPi; screen illumination fallback on mobile |
| No per-driver calibration | Different people have different baseline EAR and blink rates | 10-second calibration routine on first use |
| Binary alarm (on/off) | Alert fatigue — driver disables system entirely | 4-level progressive alerts with escalation logic |
| Phone overheating | Thermal throttling degrades accuracy after 30-60 min | Reduce to 15 FPS on mobile; RPi has no thermal constraints |
| Phone battery drain | Phone dies on long drives even while plugged in | Dedicated RPi hardware with car power; mobile as secondary |
| Mount instability | Camera angle shifts, face leaves frame | Dedicated RPi mount; detect face loss and re-prompt |

---

## Technology Choices (2026)

### Why MediaPipe Over Google ML Kit

| | Google ML Kit (2023) | MediaPipe Tasks (2026) |
|---|---|---|
| Landmarks | 468 | 478 (includes iris) |
| Blendshapes | Limited | 52 ARKit-compatible (`eyeBlinkLeft`, `jawOpen`, etc.) |
| Platforms | Android/iOS only | Android, iOS, Web, Python, RPi |
| RPi support | None | Yes (Python) |
| Model control | Black box | Configurable |
| Night/IR | No | Works on IR images |

MediaPipe is the clear winner — same model family on both platforms, provides all drowsiness signals in a single inference pass.

### Alternative Models (If MediaPipe Is Too Heavy)

| Model | Landmarks | Size | RPi 5 FPS | Use Case |
|-------|-----------|------|-----------|----------|
| MediaPipe Face Landmarker | 478 | ~2MB | 20-30 (CPU) | Primary choice |
| PFLD | 98 | ~1MB | 40-55 (CPU) | Lightweight fallback |
| YuNet (face detection only) | 5 key | ~350KB | 20-30 | Pair with PFLD |
| dlib 68-landmark | 68 | ~100MB | 5-10 | Avoid — too slow |

Resource: `PINTO0309/PINTO_model_zoo` on GitHub has pre-converted edge-optimized models in TFLite, ONNX, and NCNN formats.

---

## Drowsiness Detection Algorithm

### Multi-Signal Approach

#### 1. PERCLOS (Primary — Weight: 0.35)

The gold standard metric validated by NHTSA/FHWA research.

- Compute Eye Aspect Ratio (EAR) per frame:
  ```
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  ```
- Eye is "closed" when EAR < calibrated threshold (typically ~0.20)
- PERCLOS = count(closed frames) / total(frames) over a 60-second rolling window
- Alert threshold: PERCLOS > 0.15

#### 2. Blink Duration Analysis (Weight: 0.25)

- Normal blink: 100-400ms
- Drowsy blink: >500ms
- Track EAR transitions — a blink is a dip below threshold and recovery
- Measure duration of each dip

#### 3. Yawn Detection (Weight: 0.15)

- Mouth Aspect Ratio (MAR) from mouth landmarks:
  ```
  MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
  ```
- Yawn = MAR > 0.6 sustained for >2 seconds
- 3+ yawns in 5 minutes = drowsiness signal
- Higher false positive rate (talking, singing) — never use alone

#### 4. Head Pose Estimation (Weight: 0.15)

- Use `cv2.solvePnP` with 3D face model and 2D landmark projections
- Track pitch (nodding), yaw (looking away), roll (tilt)
- Head nod = pitch drops >15 degrees below calibrated baseline then jerks back
- Head drooping = sustained downward pitch

#### 5. Gaze Direction (Weight: 0.10)

- MediaPipe iris landmarks enable monocular gaze estimation (~5-7 degree accuracy)
- If gaze deviates from forward for >2-3 seconds = inattention alert
- Detects phone use, looking at lap

### Composite Score

```
drowsiness_score = 0.35 * normalize(PERCLOS, 0, 0.4)
                 + 0.25 * normalize(avg_blink_duration, 0, 800ms)
                 + 0.15 * normalize(yawn_frequency, 0, 6 per 5min)
                 + 0.15 * normalize(head_nod_frequency, 0, 3 per min)
                 + 0.10 * normalize(gaze_deviation, 0, 1)
```

### State Machine

```
ALERT ──→ MILD ──→ MODERATE ──→ SEVERE
  ^                                 |
  └──── all metrics baseline 30s ───┘

ALERT → MILD:      PERCLOS > 0.15 OR yawn_count > 3/5min OR EAR trend declining
MILD → MODERATE:   2+ signals active OR PERCLOS > 0.20
MODERATE → SEVERE: PERCLOS > 0.30 OR head nod detected OR eyes closed > 2s
Any → ALERT:       All metrics return to baseline for 30+ seconds
```

Design principles:
- Hysteresis: require persistence before escalating (prevents noise flicker)
- Asymmetric transitions: escalation is fast (safety), de-escalation is slow (conservative)
- Escalation memory: if MODERATE reached 3 times in 15 min, lower thresholds

### Alert Escalation

| Level | Response | Trigger |
|-------|----------|---------|
| MILD | Soft chime + yellow indicator | First threshold crossed |
| MODERATE | Louder alarm + voice: "You seem drowsy, consider a break" + vibration (mobile) | 10s after MILD or 2+ signals |
| SEVERE | Continuous alarm + "Pull over safely NOW" + cannot silence without explicit action | Immediate on critical signals |

Alert fatigue prevention:
- After driver acknowledges MILD/MODERATE, suppress same-level alerts for 2 min
- If 3+ MODERATE alerts in 15 min, escalate base sensitivity
- "I'm taking a break" button pauses monitoring for 15 min

### Per-Driver Calibration

On first launch / startup:
1. Ask driver to look at camera naturally for 10 seconds
2. Record baseline EAR mean and standard deviation
3. Set closed threshold = baseline_EAR * 0.6 (or baseline - 2 * std)
4. Optionally calibrate MAR baseline and neutral head pose
5. Store calibration profile locally

---

## Architecture

### Shared Spec, Separate Implementations

Do NOT try to share code between Flutter (Dart) and RPi (Python). Instead:

- Write a detailed algorithm specification (pseudocode + math) as the shared contract
- Implement independently on each platform using idiomatic code
- Test both against the same test vectors to verify equivalence
- The algorithms (EAR, MAR, PERCLOS, state machine) are straightforward math — divergence risk is low

The scoring/alerting logic is NOT ML — it is signal processing on time series of facial features. Keep it separate from the vision pipeline for testability.

### Platform-Specific Plans

- See `PLAN_MOBILE.md` for Flutter/Android detailed plan
- See `PLAN_RPI.md` for Raspberry Pi detailed plan

---

## Hardware: Raspberry Pi Is the Best Option

### Why RPi 5

| Criteria | RPi 5 + Hailo-8L | Orange Pi 5 (built-in NPU) | Jetson Orin Nano |
|----------|-------------------|----------------------------|------------------|
| Total system cost | ~$225 | ~$160 | ~$325 |
| Face landmark FPS | 30-60 | 25-40 | 60+ |
| Power draw | 5-8W | 5-10W | 7-15W |
| Ecosystem/docs | Excellent | Moderate | Good |
| Long-term support | Excellent | Uncertain | Good |

The RPi 5's ecosystem advantage is decisive: official camera support, vast community, extensive drowsiness detection tutorials, and official Hailo integration from the Raspberry Pi Foundation.

### Budget

| Configuration | Total | Notes |
|---------------|-------|-------|
| Minimum viable (no accelerator) | $145-165 | 10-18 FPS, marginal but works |
| Recommended (Hailo-8L AI Kit) | $215-235 | 30+ FPS, comfortable headroom |
| Full featured (+ display) | $225-250 | Recommended + status OLED |

Full bill of materials in `PLAN_RPI.md`.

---

## Project Phases

### Phase 1: Research & Foundation (Weeks 1-2)

- Benchmark MediaPipe Face Landmarker on RPi 5 (CPU-only and with Hailo)
- Benchmark MediaPipe / ML Kit on target Android device
- Write and finalize algorithm specification document (EAR, PERCLOS, MAR, head pose, state machine)
- Set up dev environments for both platforms
- Order RPi hardware
- Decision by end of Week 2: MediaPipe vs PFLD on RPi, ML Kit vs raw MediaPipe on mobile

### Phase 2: Mobile App Improvement (Weeks 3-5)

- Replace binary eye detection with EAR + PERCLOS
- Add per-driver calibration
- Add yawn detection (MAR)
- Add head pose estimation
- Implement state machine + weighted scoring
- Progressive alert system
- UI overhaul: drowsiness gauge, session stats, dark mode, settings
- Performance: target 15 FPS at 640x480

See `PLAN_MOBILE.md` for detailed breakdown.

### Phase 3: Raspberry Pi Prototype (Weeks 5-8)

Overlaps with end of Phase 2 — intentional parallelization.

- Hardware assembly: NoIR camera + IR LEDs + speaker + mount
- Camera pipeline with picamera2
- Port detection algorithm to Python
- Validate IR night driving
- Audio alerts + TTS
- Auto-start via systemd + watchdog
- Data logging

See `PLAN_RPI.md` for detailed breakdown.

### Phase 4: Testing & Optimization (Weeks 8-10)

- Controlled testing with 5-10 volunteers (diverse faces, glasses, facial hair)
- Test matrix:

| Scenario | Expected | Platform |
|----------|----------|----------|
| Normal driving, daylight | No alerts | Both |
| Normal driving, night | No alerts | Pi |
| Simulated drowsy (slow blinks) | MILD → MODERATE | Both |
| Microsleep (rapid closure) | SEVERE within 3s | Both |
| Yawning while alert | MILD after 3+ | Both |
| Head nod | MODERATE/SEVERE | Both |
| Sunglasses | Graceful degradation + warning | Both |
| Complete darkness | Normal operation (IR) | Pi |

- **SAFETY: Never test with actual drowsy driving.** Use a passenger to operate the system while an alert driver drives.
- Tune thresholds based on collected data
- Latency target: <500ms from eye closure to alert
- Edge cases: sunglasses (fall back to yawn + head pose), multiple faces, partial occlusion

### Phase 5: Polish & Documentation (Weeks 10-12)

- Mobile: onboarding flow, battery optimization, crash reporting
- Pi: auto-boot, status LED, clean shutdown, pre-configured SD card image
- Thermal soak test (mobile 2+ hrs, Pi 4+ hrs continuous)
- Documentation: build guide with photos, algorithm spec with final tuned parameters
- Results report: accuracy metrics vs 2023 baseline

---

## Risk Analysis

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MediaPipe too slow on RPi 5 (CPU-only) | Medium | High | Add Hailo-8L; fall back to PFLD; reduce resolution |
| IR glare on glasses | High | Medium | Test 850nm vs 940nm; adjust LED angle; detect glasses and warn |
| Phone thermal throttling | High | High | Reduce to 15 FPS; lower resolution; RPi is the primary platform |
| False detections from vibration/bumps | Medium | Medium | Temporal smoothing; require sustained signals |
| Camera cable vibration disconnect (Pi) | Medium | High | Secure with tape/clip; detect camera failure and alert |

### Safety Risks (CRITICAL)

| Risk | Impact | Mitigation |
|------|--------|------------|
| False negative: misses actual drowsiness | FATAL | Multi-metric approach; mandatory disclaimer |
| False positive: unnecessary alarms | Driver disables system | Calibration; progressive alerts; sensitivity tuning |
| Driver over-relies on system | FATAL | Disclaimer on every startup: "This is an aid, NOT a safety device" |
| Alarm startles driver | Accident | Progressive escalation; never jump from silence to max alarm |

### Mandatory Disclaimer (Display on Every Launch)

> "This system is a supplementary aid only. It is NOT a certified safety device and may fail to detect drowsiness. Never drive when tired. If you feel drowsy, pull over immediately."

---

## Success Metrics

| Metric | 2023 (Estimated) | 2.0 Target |
|--------|-------------------|------------|
| False positive rate | ~50%+ | <10% |
| Drowsiness indicators tracked | 1 (eye open/closed) | 4 (PERCLOS, yawn, head pose, blink) |
| Night driving support | None | Full (RPi with IR) |
| Alert levels | 1 (binary) | 4 (progressive) |
| Per-driver calibration | None | Yes |
| Continuous operation | ~30-60 min (mobile thermal) | 2+ hrs (mobile), 8+ hrs (Pi) |
| Detection → alert latency | Unknown | <500ms |
| True positive rate (sensitivity) | Unknown | >85% |
| True negative rate (specificity) | Unknown | >90% |

---

## Open Questions to Resolve in Phase 1

1. MediaPipe vs PFLD on RPi 5 — benchmark both, decide by end of Week 2
2. IR wavelength: 850nm (faint red glow) vs 940nm (invisible, cameras less sensitive) — test both
3. Glasses handling: how much IR reflects off different lens types — empirical testing needed
4. Should we log raw metrics (EAR, MAR, head angles) for future custom ML training? (Recommend: yes)
5. Multi-driver profiles via face recognition — defer to stretch goal
