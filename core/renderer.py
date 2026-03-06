import cv2
import numpy as np

_ema_state = {}
_EMA_ALPHA = 0.4


def _ema(key, value):
    if key not in _ema_state:
        _ema_state[key] = value
    else:
        _ema_state[key] = _EMA_ALPHA * value + (1 - _EMA_ALPHA) * _ema_state[key]
    return _ema_state[key]


def overlay_asset(frame, asset_img, cx, ty, face_width, angle, scale):
    w = int(face_width * scale)
    h = int(w * asset_img.shape[0] / asset_img.shape[1])
    if w <= 0 or h <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (w, h))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (w, h))

    x1, y1 = cx - w // 2, ty - h
    x2, y2 = x1 + w, y1 + h

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def _lm(landmarks, idx, fw, fh):
    lm = landmarks[idx]
    return int(lm.x * fw), int(lm.y * fh)


def _stable_angle(landmarks, fw, fh):
    lx, ly = _lm(landmarks, 234, fw, fh)
    rx, ry = _lm(landmarks, 454, fw, fh)
    return -np.degrees(np.arctan2(ry - ly, rx - lx))


def apply_hat(frame, landmarks, asset_img, scale=1.4):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 234, fw, fh)
    rx, ry = _lm(landmarks, 454, fw, fh)
    tx, ty = _lm(landmarks, 10,  fw, fh)

    face_width = int(np.linalg.norm([rx - lx, ry - ly]))
    raw_cx = (lx + rx) // 2
    raw_ty = ty - int(face_width * 0.05)
    raw_angle = _stable_angle(landmarks, fw, fh)
    raw_fw = face_width

    cx    = int(_ema("hat_cx", raw_cx))
    ty_s  = int(_ema("hat_ty", raw_ty))
    angle = float(_ema("hat_angle", raw_angle))
    fw_s  = int(_ema("hat_fw", raw_fw))

    return overlay_asset(frame, asset_img, cx, ty_s, fw_s, angle, scale)


def apply_mustache(frame, landmarks, asset_img, scale=1.5):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 61,  fw, fh)
    rx, ry = _lm(landmarks, 291, fw, fh)
    _, ny  = _lm(landmarks, 164, fw, fh)

    mouth_width = int(np.linalg.norm([rx - lx, ry - ly]))
    raw_angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    raw_cx = (lx + rx) // 2
    raw_cy = ny

    cx    = int(_ema("mu_cx", raw_cx))
    cy    = int(_ema("mu_cy", raw_cy))
    angle = float(_ema("mu_angle", raw_angle))
    mw_s  = int(_ema("mu_mw", mouth_width))

    mw = int(mw_s * scale)
    mh = int(mw * asset_img.shape[0] / asset_img.shape[1])
    if mw <= 0 or mh <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (mw, mh))
    M = cv2.getRotationMatrix2D((mw // 2, mh // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (mw, mh))

    x1, y1 = cx - mw // 2, cy - mh // 2
    x2, y2 = x1 + mw, y1 + mh

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def apply_gif(frame, landmarks, gif_frames, frame_idx, scale=1.8):
    asset_img = gif_frames[frame_idx % len(gif_frames)]
    return apply_hat(frame, landmarks, asset_img, scale=scale)
