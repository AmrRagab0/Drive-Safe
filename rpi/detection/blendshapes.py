"""Extract drowsiness signals from MediaPipe FaceLandmarker blendshapes.

Blendshape scores are 0.0-1.0 outputs from a learned head on top of the face
landmarks. They are the open-source equivalent of Google ML Kit's
`eyeOpenProbability` / `jawOpen` outputs.

Relevant blendshapes for drowsiness:
- `eyeBlinkLeft`, `eyeBlinkRight`: 0.0 = fully open, 1.0 = fully closed.
- `jawOpen`: 0.0 = closed, 1.0 = fully open (yawn signal).
"""

EYE_BLINK_LEFT = "eyeBlinkLeft"
EYE_BLINK_RIGHT = "eyeBlinkRight"
JAW_OPEN = "jawOpen"


def eye_closure_score(blendshapes):
    """Per-frame eye-closure probability in [0, 1].

    Takes max across both eyes so a single closed eye still triggers — matches
    ML Kit's behaviour and is robust to one eye being occluded.

    Args:
        blendshapes: dict mapping blendshape name -> score, or None.

    Returns:
        float in [0, 1]. Returns 0.0 if blendshapes unavailable.
    """
    if not blendshapes:
        return 0.0
    return max(
        blendshapes.get(EYE_BLINK_LEFT, 0.0),
        blendshapes.get(EYE_BLINK_RIGHT, 0.0),
    )


def jaw_open_score(blendshapes):
    """Per-frame jaw-open probability in [0, 1].

    Args:
        blendshapes: dict mapping blendshape name -> score, or None.

    Returns:
        float in [0, 1]. Returns 0.0 if blendshapes unavailable.
    """
    if not blendshapes:
        return 0.0
    return blendshapes.get(JAW_OPEN, 0.0)
