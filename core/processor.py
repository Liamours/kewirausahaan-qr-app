import av
import cv2
import threading
from core.facemesh import FaceMesh
from core.renderer import apply_hat
from streamlit_webrtc import VideoProcessorBase


class FaceFilterProcessor(VideoProcessorBase):
    def __init__(self, hat_asset):
        self.face_mesh = FaceMesh()
        self.hat_asset = hat_asset
        self._lock = threading.Lock()
        self._snapshot = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                img = apply_hat(img, landmarks, self.hat_asset)

        with self._lock:
            self._snapshot = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self):
        with self._lock:
            return self._snapshot