"""
Face detection via OpenCV Haar cascade.

No native shared-library dependencies, no model downloads — works on
every platform opencv-python-headless supports, including Python 3.13.

Public interface: FaceMesh().process(rgb_frame) → _DetectionResult
  result.face_landmarks  →  list of _FaceLandmarks
  landmarks[idx].x / .y →  normalised float in [0, 1]
"""

import os
import cv2


class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FaceLandmarks:
    """
    Six key-point estimates derived from a face bounding box.

    Fractions are chosen to match the approximate positions of the
    MediaPipe 468-point indices that renderer.py depends on:
        10  → forehead top-centre
        234 → left cheek edge
        454 → right cheek edge
        61  → left mouth corner
        291 → right mouth corner
        164 → philtrum (just below nose)
    """

    _FRACS: dict = {
        10:  (0.50, 0.02),
        234: (0.00, 0.45),
        454: (1.00, 0.45),
        61:  (0.28, 0.75),
        291: (0.72, 0.75),
        164: (0.50, 0.68),
    }

    def __init__(self, x: int, y: int, w: int, h: int,
                 img_w: int, img_h: int) -> None:
        self._lm: dict = {
            idx: _Landmark((x + w * fx) / img_w, (y + h * fy) / img_h)
            for idx, (fx, fy) in self._FRACS.items()
        }

    def __getitem__(self, idx: int) -> _Landmark:
        return self._lm[idx]


class _DetectionResult:
    __slots__ = ("face_landmarks",)

    def __init__(self, face_landmarks: list) -> None:
        self.face_landmarks = face_landmarks


class FaceMesh:
    def __init__(self, max_faces: int = 2,
                 min_detect: float = 0.5,
                 min_track: float = 0.5) -> None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from: {cascade_path}"
            )
        self._max_faces = max_faces

    def process(self, rgb_frame) -> _DetectionResult:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        landmarks: list = []
        if len(faces) > 0:
            for (x, y, w, h) in faces[: self._max_faces]:
                landmarks.append(_FaceLandmarks(x, y, w, h, img_w, img_h))

        return _DetectionResult(landmarks)

    def close(self) -> None:
        pass
