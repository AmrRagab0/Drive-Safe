import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:audioplayers/audioplayers.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(
      this.absoluteImageSize, this.faces, this.playSound, this.stopSound);
  int colorInt = 1;
  final Size absoluteImageSize;
  final List<Face> faces;
  final Function playSound;
  final Function stopSound;

  List<Color> colors = [
    Colors.red,
    Colors.green,
    Colors.indigo,
    Colors.limeAccent,
    Colors.orange
  ];

  @override
  void paint(Canvas canvas, Size size) async {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;

    try {
      Face face = faces[0];

      double averageEyeOpenProb =
          (face.leftEyeOpenProbability! + face.rightEyeOpenProbability!) / 2.0;

      if (averageEyeOpenProb < 0.6) {
        print("Alert");

        playSound();
        colorInt = 0;
      } else {
        stopSound();
        colorInt = 1;
      }

      canvas.drawRect(
          Rect.fromLTRB(
              face.boundingBox.left * scaleX,
              face.boundingBox.top * scaleY,
              face.boundingBox.right * scaleX,
              face.boundingBox.bottom * scaleY),
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 6.0
            ..color = colors[colorInt]);
    } catch (e) {
      print("Can't detect your face");
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.absoluteImageSize != absoluteImageSize ||
        oldDelegate.faces != faces;
  }
}
