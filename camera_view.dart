import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'painter.dart';
import 'package:audioplayers/audioplayers.dart';

import 'utils.dart';

class CameraPreviewScanner extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _CameraPreviewScannerState();
}

class _CameraPreviewScannerState extends State<CameraPreviewScanner> {
  dynamic _scanResults;
  late CameraController _camera;
  bool _isDetecting = false;
  CameraLensDirection _direction = CameraLensDirection.front;
  AudioPlayer audioPlayer = AudioPlayer()..setReleaseMode(ReleaseMode.loop);

  bool isSoundPlaying = true;
  void playAlertSound() async {
    if (!isSoundPlaying) {
      audioPlayer.play(AssetSource('beeping.mp3'));
      isSoundPlaying = true;
    }
  }

  void stopAlertSound() {
    if (isSoundPlaying) {
      audioPlayer.stop();
      isSoundPlaying = false;
    }
  }

  final FaceDetector _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
    performanceMode: FaceDetectorMode.accurate,
    enableLandmarks: true,
    enableContours: true,
    enableClassification: true,
    enableTracking: true,
  ));
  @override
  void initState() {
    super.initState();

    _initializeCamera();
  }

  void _initializeCamera() async {
    final CameraDescription description =
        await ScannerUtils.getCamera(_direction);
    setState(() {});
    _camera = CameraController(
      description,
      defaultTargetPlatform == TargetPlatform.android
          ? ResolutionPreset.veryHigh
          : ResolutionPreset.high,
    );
    await _camera.initialize().catchError((onError) => print(onError));

    _camera.startImageStream((CameraImage image) {
      if (_isDetecting) return;

      _isDetecting = true;

      ScannerUtils.detect(
        image: image,
        detectInImage: _faceDetector.processImage,
        imageRotation: description.sensorOrientation,
      ).then(
        (dynamic results) {
          setState(() {
            _scanResults = results;
          });
        },
      ).whenComplete(() => _isDetecting = false);
    });
  }

  Widget _buildResults() {
    Text noResultsText = Text("Can't find your face");

    if (_scanResults == null ||
        _camera == null ||
        !_camera.value.isInitialized) {
      return noResultsText;
    }

    final Size imageSize = Size(
      _camera.value.previewSize!.height,
      _camera.value.previewSize!.width,
    );
    if (_scanResults is! List<Face>) return noResultsText;

    CustomPainter? painter = FaceDetectorPainter(
        imageSize,
        _scanResults.cast<Face>(),
        playAlertSound,
        stopAlertSound) as CustomPainter?;

    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    return Container(
      constraints: BoxConstraints.expand(),
      color: const Color.fromARGB(26, 223, 221, 221),
      child: _camera == null
          ? Center(
              child: Text(
                'Initializing Camera...',
                style: TextStyle(
                  color: Colors.red,
                  fontSize: 30.0,
                ),
              ),
            )
          : Padding(
              padding: const EdgeInsets.all(0.1),
              child: Stack(
                fit: StackFit.expand,
                children: <Widget>[
                  CameraPreview(_camera),
                  _buildResults(),
                  // Text(face)
                ],
              ),
            ),
    );
  }

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
    } else {
      _direction = CameraLensDirection.back;
    }

    await _camera.stopImageStream();
    await _camera.dispose();

    setState(() {
      //_camera =  ;
    });

    _initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: Text(
            'Drive Safe',
          ),
        ),
        backgroundColor: Colors.black,
        //?centerTitle: true,
      ),
      body: _buildImage(),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: <Widget>[
          SizedBox(
            width: 25,
          ),
          FloatingActionButton(
            foregroundColor: Colors.white,
            backgroundColor: Colors.black,
            onPressed: _toggleCameraDirection,
            child: _direction == CameraLensDirection.front
                ? Icon(Icons.camera_rear)
                : Icon(Icons.camera_front),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _camera.dispose().then((_) {
      _faceDetector.close();
    });

    super.dispose();
  }
}
