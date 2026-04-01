# Drive Safe 2.0 — Mobile App (Flutter) Detailed Plan

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Framework | Flutter 3.x (Dart) | Cross-platform, existing codebase |
| Face Detection | google_mlkit_face_detection (wraps MediaPipe) | 468+ landmarks, blendshapes |
| Camera | camera plugin | Real-time preview + frame access |
| Audio Alerts | just_audio or audioplayers | Sound playback |
| Text-to-Speech | flutter_tts | Voice warnings |
| State Management | Riverpod or Provider | Reactive UI updates |
| Local Storage | shared_preferences + sqflite | Calibration profiles + session logs |
| Background Processing | Isolates | Offload inference from UI thread |

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection FPS | 15 FPS | Processing every 2nd frame at 30fps capture |
| Capture resolution | 640x480 | Lower than camera native — saves CPU/battery |
| End-to-end latency | <100ms per frame | Inference + scoring |
| Detection → alert | <500ms | Eye closure to alarm sound |
| Continuous operation | 2+ hours | Without thermal throttling degrading accuracy |
| Battery impact | <20% per hour (plugged in = net positive) | Aggressive power management |

---

## Known Limitations (Mobile Platform)

### Night Driving

Phone front cameras have IR filters — they are nearly useless in the dark. Mitigations:

1. **Screen illumination**: Use a dim warm-toned screen as a light source. Not ideal but provides some face illumination.
2. **Ambient light detection**: If light level drops below threshold, warn user that accuracy is reduced.
3. **Recommend RPi version**: For regular night driving, the RPi with IR camera is the proper solution. The mobile app should display a recommendation.

The mobile app is best suited for daytime and well-lit nighttime driving (city streets, lit highways). For dark rural roads or highways without lighting, the RPi version is necessary.

### Thermal Management

Continuous camera + ML inference will heat the phone. Strategy:

- Process at 15 FPS, not 30 (skip every other frame)
- Use 640x480, not full camera resolution
- Monitor CPU temperature if platform API allows
- If thermal throttling detected: reduce to 10 FPS and warn user
- Suggest user position phone near A/C vent
- Display temperature warning if device is overheating

---

## Development Plan (Weeks 3-5)

### Week 3: Core Detection Upgrade

#### Task 3.1: Replace Binary Eye Detection with EAR

The existing app detects "eyes open" or "eyes closed" as a binary. Replace with continuous Eye Aspect Ratio.

```dart
// Eye Aspect Ratio calculation from 6 landmarks per eye
double computeEAR(List<Point<double>> eyeLandmarks) {
  // p1=lateral, p2=upper-outer, p3=upper-inner,
  // p4=medial, p5=lower-inner, p6=lower-outer
  final p1 = eyeLandmarks[0];
  final p2 = eyeLandmarks[1];
  final p3 = eyeLandmarks[2];
  final p4 = eyeLandmarks[3];
  final p5 = eyeLandmarks[4];
  final p6 = eyeLandmarks[5];

  double vertical1 = _distance(p2, p6);
  double vertical2 = _distance(p3, p5);
  double horizontal = _distance(p1, p4);

  return (vertical1 + vertical2) / (2.0 * horizontal);
}

double _distance(Point<double> a, Point<double> b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
```

If using ML Kit's face detection with classification mode, the `leftEyeOpenProbability` and `rightEyeOpenProbability` values (0.0-1.0) can be used as a simpler proxy for EAR. However, raw landmark-based EAR gives more control and precision.

#### Task 3.2: Implement PERCLOS

```dart
class PERCLOSCalculator {
  final int windowSize; // number of frames in rolling window
  final Queue<bool> _closedFrames = Queue();

  PERCLOSCalculator({required int fps, int windowSeconds = 60})
      : windowSize = fps * windowSeconds;

  double update(double ear, double closedThreshold) {
    bool isClosed = ear < closedThreshold;
    _closedFrames.addLast(isClosed);

    while (_closedFrames.length > windowSize) {
      _closedFrames.removeFirst();
    }

    if (_closedFrames.isEmpty) return 0;
    int closedCount = _closedFrames.where((c) => c).length;
    return closedCount / _closedFrames.length;
  }

  void reset() => _closedFrames.clear();
}
```

#### Task 3.3: Implement Per-Driver Calibration

```dart
class CalibrationResult {
  final double earBaseline;
  final double earStdDev;
  final double earClosedThreshold;
  final double marBaseline;
  final double pitchBaseline;

  CalibrationResult({
    required this.earBaseline,
    required this.earStdDev,
    required this.earClosedThreshold,
    required this.marBaseline,
    required this.pitchBaseline,
  });

  Map<String, dynamic> toJson() => { /* serialize for storage */ };

  factory CalibrationResult.fromJson(Map<String, dynamic> json) { /* deserialize */ }
}

class Calibrator {
  static const duration = Duration(seconds: 10);

  Future<CalibrationResult> calibrate(Stream<FaceData> faceStream) async {
    final earSamples = <double>[];
    final marSamples = <double>[];
    final pitchSamples = <double>[];

    final completer = Completer<CalibrationResult>();

    Timer(duration, () {
      double earMean = earSamples.average;
      double earStd = earSamples.standardDeviation;

      completer.complete(CalibrationResult(
        earBaseline: earMean,
        earStdDev: earStd,
        earClosedThreshold: max(earMean - 2 * earStd, earMean * 0.6),
        marBaseline: marSamples.average,
        pitchBaseline: pitchSamples.average,
      ));
    });

    faceStream.listen((data) {
      earSamples.add(data.ear);
      marSamples.add(data.mar);
      pitchSamples.add(data.pitch);
    });

    return completer.future;
  }
}
```

- Store calibration with `shared_preferences` or `sqflite`
- Allow re-calibration from settings
- Show calibration screen on first launch with voice prompt: "Look at the camera naturally for 10 seconds"

#### Task 3.4: Implement Yawn Detection

```dart
double computeMAR(List<Point<double>> mouthLandmarks) {
  // Upper/lower lip landmarks for vertical distances
  // Corner landmarks for horizontal distance
  final p1 = mouthLandmarks[0]; // left corner
  final p2 = mouthLandmarks[1]; // upper outer
  final p3 = mouthLandmarks[2]; // upper inner
  final p4 = mouthLandmarks[3]; // right corner (or use index appropriately)
  final p5 = mouthLandmarks[4]; // lower inner
  final p6 = mouthLandmarks[5]; // lower outer
  // Simplified MAR
  double vertical1 = _distance(p2, p6);
  double vertical2 = _distance(p3, p5);
  double horizontal = _distance(p1, p4);

  return (vertical1 + vertical2) / (2.0 * horizontal);
}

class YawnDetector {
  final double marThreshold;
  final Duration minYawnDuration = const Duration(seconds: 2);
  DateTime? _yawnStart;
  final List<DateTime> _yawnTimestamps = [];

  YawnDetector({this.marThreshold = 0.6});

  bool update(double mar) {
    bool mouthWideOpen = mar > marThreshold;

    if (mouthWideOpen && _yawnStart == null) {
      _yawnStart = DateTime.now();
    } else if (!mouthWideOpen && _yawnStart != null) {
      if (DateTime.now().difference(_yawnStart!) >= minYawnDuration) {
        _yawnTimestamps.add(DateTime.now());
        // Remove yawns older than 5 minutes
        _yawnTimestamps.removeWhere(
          (t) => DateTime.now().difference(t) > const Duration(minutes: 5)
        );
      }
      _yawnStart = null;
    }

    return mouthWideOpen && _yawnStart != null &&
           DateTime.now().difference(_yawnStart!) >= minYawnDuration;
  }

  int get yawnCountLast5Min => _yawnTimestamps.length;
}
```

### Week 4: Head Pose, State Machine & Scoring

#### Task 4.1: Head Pose Estimation

For ML Kit, head pose (Euler angles) is available directly from the `Face` object:

```dart
void processHeadPose(Face face) {
  final pitch = face.headEulerAngleX; // nodding (up/down)
  final yaw = face.headEulerAngleY;   // turning (left/right)
  final roll = face.headEulerAngleZ;  // tilting (ear to shoulder)

  // Detect head nod: pitch drops below baseline - 15 degrees
  bool headNod = pitch != null &&
                 pitch < (calibration.pitchBaseline - 15);

  // Detect looking away: yaw exceeds 30 degrees
  bool lookingAway = yaw != null && yaw.abs() > 30;
}
```

If using raw MediaPipe landmarks, use `solvePnP` as described in the RPi plan (via a platform channel to native code or a Dart implementation).

#### Task 4.2: Blink Duration Tracker

```dart
class BlinkTracker {
  double earThreshold;
  DateTime? _blinkStart;
  final List<double> _blinkDurations = []; // in seconds

  BlinkTracker({required this.earThreshold});

  void update(double ear) {
    bool eyesClosed = ear < earThreshold;

    if (eyesClosed && _blinkStart == null) {
      _blinkStart = DateTime.now();
    } else if (!eyesClosed && _blinkStart != null) {
      double duration = DateTime.now().difference(_blinkStart!).inMilliseconds / 1000.0;
      _blinkDurations.add(duration);
      if (_blinkDurations.length > 100) _blinkDurations.removeAt(0);
      _blinkStart = null;
    }
  }

  double get avgBlinkDuration {
    if (_blinkDurations.isEmpty) return 0;
    final recent = _blinkDurations.length > 10
        ? _blinkDurations.sublist(_blinkDurations.length - 10)
        : _blinkDurations;
    return recent.reduce((a, b) => a + b) / recent.length;
  }

  /// Duration of current ongoing eye closure (0 if eyes are open)
  double get currentClosureDuration {
    if (_blinkStart == null) return 0;
    return DateTime.now().difference(_blinkStart!).inMilliseconds / 1000.0;
  }
}
```

#### Task 4.3: Drowsiness State Machine

```dart
enum DrowsinessState { alert, mild, moderate, severe }

class DrowsinessStateMachine {
  DrowsinessState _state = DrowsinessState.alert;
  DateTime _stateEntryTime = DateTime.now();
  int _moderateCountIn15Min = 0;
  final List<DateTime> _moderateTimestamps = [];

  DrowsinessState get state => _state;

  DrowsinessState update({
    required double perclos,
    required double avgBlinkDuration,
    required int yawnCount5Min,
    required bool headNod,
    required double eyesClosedDuration,
  }) {
    switch (_state) {
      case DrowsinessState.alert:
        if (perclos > 0.15 || yawnCount5Min > 3) {
          _transition(DrowsinessState.mild);
        }
        break;

      case DrowsinessState.mild:
        int activeSignals = [
          perclos > 0.15,
          avgBlinkDuration > 0.5,
          yawnCount5Min > 3,
          headNod,
        ].where((s) => s).length;

        if (activeSignals >= 2 || perclos > 0.20) {
          _transition(DrowsinessState.moderate);
        } else if (_secondsInState > 30 && perclos < 0.10 && !headNod && yawnCount5Min < 2) {
          _transition(DrowsinessState.alert);
        }
        break;

      case DrowsinessState.moderate:
        if (perclos > 0.30 || headNod || eyesClosedDuration > 2.0) {
          _transition(DrowsinessState.severe);
        } else if (_secondsInState > 30 && perclos < 0.10 && !headNod) {
          _transition(DrowsinessState.alert);
        }
        break;

      case DrowsinessState.severe:
        if (_secondsInState > 30 && perclos < 0.10 && !headNod && eyesClosedDuration < 0.5) {
          _transition(DrowsinessState.moderate);
        }
        break;
    }

    return _state;
  }

  void _transition(DrowsinessState newState) {
    if (newState == DrowsinessState.moderate) {
      _moderateTimestamps.add(DateTime.now());
      _moderateTimestamps.removeWhere(
        (t) => DateTime.now().difference(t) > const Duration(minutes: 15)
      );
    }
    _state = newState;
    _stateEntryTime = DateTime.now();
  }

  double get _secondsInState =>
      DateTime.now().difference(_stateEntryTime).inSeconds.toDouble();

  int get moderateCountLast15Min => _moderateTimestamps.length;
}
```

#### Task 4.4: Weighted Drowsiness Score

```dart
class DrowsinessScorer {
  static double compute({
    required double perclos,
    required double avgBlinkDuration,
    required int yawnCount5Min,
    required double headNodFrequency, // nods per minute
    required double gazeDeviation,    // 0-1 normalized
  }) {
    return 0.35 * _normalize(perclos, 0, 0.4) +
           0.25 * _normalize(avgBlinkDuration, 0, 0.8) +
           0.15 * _normalize(yawnCount5Min.toDouble(), 0, 6) +
           0.15 * _normalize(headNodFrequency, 0, 3) +
           0.10 * _normalize(gazeDeviation, 0, 1);
  }

  static double _normalize(double value, double min, double max) {
    return ((value - min) / (max - min)).clamp(0.0, 1.0);
  }
}
```

### Week 5: Alerts, UI & Performance

#### Task 5.1: Progressive Alert System

```dart
class AlertManager {
  final FlutterTts _tts = FlutterTts();
  final AudioPlayer _player = AudioPlayer();
  final Map<DrowsinessState, DateTime> _lastAlertTimes = {};

  static const _cooldowns = {
    DrowsinessState.mild: Duration(minutes: 2),
    DrowsinessState.moderate: Duration(minutes: 1),
    DrowsinessState.severe: Duration.zero,
  };

  Future<void> handleState(DrowsinessState state) async {
    if (state == DrowsinessState.alert) {
      await _player.stop();
      return;
    }

    // Check cooldown
    final lastAlert = _lastAlertTimes[state];
    if (lastAlert != null &&
        DateTime.now().difference(lastAlert) < _cooldowns[state]!) {
      return;
    }

    switch (state) {
      case DrowsinessState.mild:
        await _player.play(AssetSource('sounds/chime_soft.mp3'));
        HapticFeedback.lightImpact();
        break;

      case DrowsinessState.moderate:
        await _player.play(AssetSource('sounds/alarm_medium.mp3'));
        HapticFeedback.heavyImpact();
        await _tts.speak("You appear drowsy. Consider taking a break.");
        break;

      case DrowsinessState.severe:
        await _player.setReleaseMode(ReleaseMode.loop);
        await _player.play(AssetSource('sounds/alarm_loud.mp3'));
        HapticFeedback.heavyImpact();
        await _tts.speak("Warning! Pull over safely now!");
        // Vibrate continuously
        break;

      default:
        break;
    }

    _lastAlertTimes[state] = DateTime.now();
  }

  Future<void> acknowledge() async {
    await _player.stop();
    // Reset to monitoring — do NOT reset state machine
  }
}
```

#### Task 5.2: UI Design

**Main Driving Screen:**
```
┌─────────────────────────────────┐
│  [Camera Preview - small]       │
│                                 │
│     ┌───────────────────┐       │
│     │  DROWSINESS GAUGE │       │
│     │   ████░░░░░░░░░   │       │
│     │    32% — ALERT     │       │
│     └───────────────────┘       │
│                                 │
│  Drive time: 1h 23m             │
│  Alerts: 0                      │
│                                 │
│  ┌─────────┐  ┌──────────────┐  │
│  │ Settings│  │ End Session  │  │
│  └─────────┘  └──────────────┘  │
└─────────────────────────────────┘
```

Key UI elements:
- **Drowsiness gauge**: horizontal bar, color-coded (green → yellow → orange → red)
- **State indicator**: large text showing current state
- **Camera preview**: small, in corner (or hidden — just for debug/positioning)
- **Drive time**: duration of current monitoring session
- **Alert count**: how many alerts triggered this session
- **Dark mode**: default ON — bright screens at night are a distraction
- **Minimal interaction required**: the app should work hands-free after starting

**Settings Screen:**
- Sensitivity slider (adjusts PERCLOS thresholds: lower = more sensitive)
- Alert sound selection
- TTS voice on/off
- Re-run calibration
- Night mode brightness control
- View session history

**Calibration Screen:**
```
┌─────────────────────────────────┐
│                                 │
│  [Camera Preview - large]       │
│  ┌─────────────────────────┐    │
│  │  Face outline overlay   │    │
│  │  "Position your face    │    │
│  │   in the circle"        │    │
│  └─────────────────────────┘    │
│                                 │
│  Look at the camera naturally   │
│  for 10 seconds...              │
│                                 │
│  ████████░░░░  7/10 seconds     │
│                                 │
└─────────────────────────────────┘
```

#### Task 5.3: Session History (sqflite)

```dart
class SessionDatabase {
  static const _tableName = 'sessions';

  Future<Database> get database async {
    return openDatabase(
      join(await getDatabasesPath(), 'drivesafe.db'),
      onCreate: (db, version) {
        return db.execute('''
          CREATE TABLE $_tableName (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            duration_minutes INTEGER,
            mild_alerts INTEGER,
            moderate_alerts INTEGER,
            severe_alerts INTEGER,
            avg_perclos REAL,
            max_perclos REAL,
            yawn_count INTEGER
          )
        ''');
      },
      version: 1,
    );
  }

  Future<void> saveSession(DrivingSession session) async {
    final db = await database;
    await db.insert(_tableName, session.toMap());
  }

  Future<List<DrivingSession>> getRecentSessions({int limit = 20}) async {
    final db = await database;
    final maps = await db.query(_tableName, orderBy: 'start_time DESC', limit: limit);
    return maps.map((m) => DrivingSession.fromMap(m)).toList();
  }
}
```

#### Task 5.4: Performance Optimization

```dart
class DetectionController {
  static const _targetFps = 15;
  bool _isProcessing = false;
  int _frameSkipCount = 0;

  void onCameraFrame(CameraImage image) {
    // Skip frames to maintain target FPS
    _frameSkipCount++;
    if (_frameSkipCount % 2 != 0) return; // process every 2nd frame

    // Don't queue frames if previous is still processing
    if (_isProcessing) return;

    _isProcessing = true;
    _processFrame(image).then((_) {
      _isProcessing = true;
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    // Run ML Kit inference on platform thread (not Dart isolate)
    // ML Kit already runs on background thread internally
    final faces = await _faceDetector.processImage(inputImage);

    if (faces.isEmpty) {
      _handleFaceLost();
      return;
    }

    final face = faces.first; // use largest/closest face

    // All these computations are pure Dart math — fast
    final ear = _computeEAR(face);
    final perclos = _perclosCalculator.update(ear, _calibration.earClosedThreshold);
    final mar = _computeMAR(face);
    _yawnDetector.update(mar);
    _blinkTracker.update(ear);

    final headNod = face.headEulerAngleX != null &&
        face.headEulerAngleX! < (_calibration.pitchBaseline - 15);

    // Update state machine
    final state = _stateMachine.update(
      perclos: perclos,
      avgBlinkDuration: _blinkTracker.avgBlinkDuration,
      yawnCount5Min: _yawnDetector.yawnCountLast5Min,
      headNod: headNod,
      eyesClosedDuration: _blinkTracker.currentClosureDuration,
    );

    // Update UI and trigger alerts
    _drowsinessNotifier.value = DrowsinessData(
      state: state,
      perclos: perclos,
      ear: ear,
      score: DrowsinessScorer.compute(...),
    );

    _alertManager.handleState(state);
  }
}
```

Camera configuration for power efficiency:
```dart
final cameras = await availableCameras();
final frontCamera = cameras.firstWhere(
  (c) => c.lensDirection == CameraLensDirection.front
);

final controller = CameraController(
  frontCamera,
  ResolutionPreset.medium, // 640x480 — not high/max
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.nv21, // fastest format for ML Kit on Android
);
```

---

## App Architecture

```
lib/
├── main.dart
├── app.dart
├── config/
│   └── thresholds.dart          # Default thresholds and weights
├── models/
│   ├── drowsiness_state.dart    # DrowsinessState enum
│   ├── calibration_result.dart  # Calibration data model
│   └── driving_session.dart     # Session log data model
├── detection/
│   ├── face_detector_service.dart  # ML Kit wrapper
│   ├── ear_calculator.dart         # Eye Aspect Ratio
│   ├── mar_calculator.dart         # Mouth Aspect Ratio
│   ├── perclos_calculator.dart     # PERCLOS rolling window
│   ├── blink_tracker.dart          # Blink duration analysis
│   ├── yawn_detector.dart          # Yawn detection
│   └── head_pose_tracker.dart      # Head pose from Euler angles
├── scoring/
│   ├── drowsiness_scorer.dart      # Weighted composite score
│   └── state_machine.dart          # 4-state drowsiness FSM
├── calibration/
│   └── calibrator.dart             # Per-driver calibration
├── alerts/
│   └── alert_manager.dart          # Progressive alert system
├── data/
│   └── session_database.dart       # sqflite session history
├── ui/
│   ├── screens/
│   │   ├── driving_screen.dart     # Main monitoring screen
│   │   ├── calibration_screen.dart # Calibration flow
│   │   ├── settings_screen.dart    # App settings
│   │   └── history_screen.dart     # Session history
│   └── widgets/
│       ├── drowsiness_gauge.dart   # Color-coded progress bar
│       ├── state_indicator.dart    # Current state display
│       └── camera_preview.dart     # Small camera preview
└── controllers/
    └── detection_controller.dart   # Main detection loop orchestrator
```

---

## Dependencies (pubspec.yaml additions)

```yaml
dependencies:
  google_mlkit_face_detection: ^0.11.0
  camera: ^0.11.0
  just_audio: ^0.9.0
  flutter_tts: ^4.0.0
  sqflite: ^2.3.0
  shared_preferences: ^2.2.0
  path_provider: ^2.1.0
  provider: ^6.1.0  # or riverpod

dev_dependencies:
  flutter_test:
    sdk: flutter
```

---

## Migration from 2023 Codebase

If the existing Flutter codebase is available:

1. **Keep**: Flutter project structure, camera setup, basic UI scaffold, ML Kit dependency
2. **Replace**: Binary eye detection logic → EAR + PERCLOS + multi-signal
3. **Add**: Calibration flow, state machine, progressive alerts, session history, settings
4. **Upgrade**: ML Kit dependency to latest version (check for breaking changes)
5. **Remove**: Simple threshold-based alarm trigger

If starting fresh (existing code is minimal/lost):
- Use the architecture above as the blueprint
- The core detection logic is ~500 lines of Dart
- UI is standard Flutter — 3-4 screens
- Total estimated effort: 2-3 weeks for one developer

---

## Testing Plan (Mobile-Specific)

### Unit Tests
- [ ] EAR calculation with known landmark positions
- [ ] PERCLOS calculation: verify rolling window behavior
- [ ] MAR calculation with known positions
- [ ] State machine transitions: all paths
- [ ] Scorer normalization: boundary values
- [ ] Blink duration tracking: verify timing accuracy
- [ ] Yawn detection: duration threshold, counting

### Integration Tests
- [ ] ML Kit returns valid landmarks for front camera
- [ ] Camera frame → detection pipeline → state update works end-to-end
- [ ] Alert sounds play correctly for each state
- [ ] TTS speaks at correct volume
- [ ] Calibration saves and loads correctly
- [ ] Session data persists across app restarts

### Manual Testing
- [ ] Normal blinking does NOT trigger alerts
- [ ] Slow eye closure (simulated drowsiness) triggers MILD → MODERATE
- [ ] Extended eye closure (>2s) triggers SEVERE
- [ ] Yawning detected accurately
- [ ] Head nodding detected
- [ ] Sunglasses: graceful degradation
- [ ] Dark mode is readable without being distracting
- [ ] App runs 2+ hours without crash or significant performance degradation
- [ ] Phone in portrait AND landscape (or lock to one orientation)
- [ ] Different phone models: test on at least 3 Android devices
