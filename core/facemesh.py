import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "/tmp/face_landmarker.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class FaceMesh:
    def __init__(self, max_faces=2, min_detect=0.5, min_track=0.5):
        download_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            min_face_detection_confidence=min_detect,
            min_face_presence_confidence=min_detect,
            min_tracking_confidence=min_track,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()
