import av
import cv2
import threading
from core.facemesh import FaceMesh
from core.renderer import apply_filter
from streamlit_webrtc import VideoProcessorBase


class FaceFilterProcessor(VideoProcessorBase):
    def __init__(self, assets):
        self.face_mesh = FaceMesh()
        self.assets = assets
        self.mode = "hat"
        self._lock = threading.Lock()
        self._snapshot = None
        self._gif_idx = 0
        self._last_landmarks = []
        self._ema_state = {}  # per-instance EMA state — no shared globals

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        try:
            img = self._process(img)
        except Exception:
            # Never crash the WebRTC session — return the raw mirrored frame
            pass

        with self._lock:
            self._snapshot = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _process(self, img):
        h, w = img.shape[:2]

        # Downscale for detection while preserving the aspect ratio so that
        # normalised landmark coordinates map correctly onto the full frame.
        scale = min(1.0, 640 / w)
        det_w, det_h = int(w * scale), int(h * scale)
        small = cv2.resize(img, (det_w, det_h))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        result = self.face_mesh.process(rgb)
        if result.face_landmarks:
            self._last_landmarks = result.face_landmarks

        for landmarks in self._last_landmarks:
            img = apply_filter(
                img, landmarks, self.mode,
                self.assets, self._gif_idx, self._ema_state,
            )

        if self.mode == "milky":
            self._gif_idx = (self._gif_idx + 1) % len(self.assets["milky"])

        return img

    def get_snapshot(self):
        with self._lock:
            return self._snapshot
