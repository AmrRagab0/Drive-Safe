import 'dart:async';
import 'dart:typed_data';
import 'dart:ui';
import 'package:camera/camera.dart';

import 'package:google_ml_kit/google_ml_kit.dart'; // Updated import
// Updated import
import 'package:flutter/foundation.dart';

class ScannerUtils {
  static Future<CameraDescription> getCamera(CameraLensDirection dir) async {
    return await availableCameras().then(
      (List<CameraDescription> cameras) => cameras.firstWhere(
        (CameraDescription camera) => camera.lensDirection == dir,
      ),
    );
  }

  static Future<dynamic> detect({
    required CameraImage image,
    required Future<dynamic> Function(InputImage image) detectInImage,
    required int imageRotation,
  }) async {
    return detectInImage(
      InputImage.fromBytes(
        metadata: InputImageMetadata(
            size: Size(image.width.toDouble(), image.height.toDouble()),
            rotation: _rotationIntToImageRotation(imageRotation),
            format: InputImageFormatValue.fromRawValue(image.format.raw) ??
                InputImageFormat.nv21,
            bytesPerRow: image.planes[0].bytesPerRow),
        bytes: _concatenatePlanes(image.planes),
      ),
    );
  }

  static Uint8List _concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  static InputImageRotation _rotationIntToImageRotation(int rotation) {
    switch (rotation) {
      case 0:
        return InputImageRotation.rotation0deg; // Updated constant
      case 90:
        return InputImageRotation.rotation90deg; // Updated constant
      case 180:
        return InputImageRotation.rotation180deg; // Updated constant
      default:
        assert(rotation == 270);
        return InputImageRotation.rotation270deg; // Updated constant
    }
  }
}
