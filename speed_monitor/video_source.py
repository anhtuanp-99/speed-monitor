"""Mở nguồn video (webcam / IP camera / file), cấp mốc thời gian cho từng frame."""
import os
import re
import time
from pathlib import Path

import cv2

from . import config

NETWORK_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")
_IP_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?(/.*)?$")


def resolve_source(text):
    """Chuyển chuỗi người dùng nhập thành nguồn video:
    '0'                 -> webcam
    '192.168.1.5'       -> http://192.168.1.5:8080/video   (cổng, đường dẫn mặc định trong config)
    '192.168.1.5:4747'  -> http://192.168.1.5:4747/video
    'http://...'        -> giữ nguyên
    'ten.mp4'           -> file (tìm trong video/)."""
    text = str(text).strip().strip('"').strip("'")
    if text.isdigit():
        return int(text)
    if text.lower().startswith(NETWORK_PREFIXES):
        return text

    match = _IP_RE.match(text)
    if match and all(int(part) <= 255 for part in match.group(1).split(".")):
        ip, port, path = match.groups()
        return f"http://{ip}:{port or config.IP_DEFAULT_PORT}{path or config.IP_DEFAULT_PATH}"

    path = Path(text)
    if not path.is_absolute():
        for base in (config.VIDEO_DIR, config.BASE_DIR):
            candidate = base / text
            if candidate.exists():
                return str(candidate)
    return str(path)


def is_file_source(source):
    return isinstance(source, str) and not source.lower().startswith(NETWORK_PREFIXES)


def open_video_capture(source):
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # giảm độ trễ khi stream
    return cap


class VideoStream:
    """Đọc frame kèm mốc thời gian (giây) dùng để tính tốc độ.

    - File video: t = (số frame đã đọc - 1) / FPS. Không phụ thuộc tốc độ máy xử lý
      và vẫn đúng khi bỏ bớt frame (skip) vì skip cũng được đếm.
    - Webcam / IP camera: t = thời gian thật (time.monotonic) lúc đọc frame, vì FPS
      của luồng mạng không ổn định.
    """

    def __init__(self, source):
        self.source = source
        self.cap = open_video_capture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        self.is_file = is_file_source(source)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self.fps = fps if 1 < fps < 240 else config.DEFAULT_FPS
        self.frames_read = 0
        self._t0 = time.monotonic()

    def read(self):
        """Trả về (ok, frame, t)."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        self.frames_read += 1
        if self.is_file:
            t = (self.frames_read - 1) / self.fps
        else:
            t = time.monotonic() - self._t0
        return True, frame, t

    def skip(self, n):
        """Bỏ qua n frame (chỉ có ý nghĩa với file). Trả về số frame đã bỏ."""
        skipped = 0
        for _ in range(max(0, n)):
            if not self.cap.grab():
                break
            self.frames_read += 1
            skipped += 1
        return skipped

    def restart(self):
        """Quay lại đầu file video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frames_read = 0
        self._t0 = time.monotonic()

    def release(self):
        self.cap.release()


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def fit_to_box(frame, max_w, max_h):
    """Thu nhỏ/phóng to vừa khung hiển thị, giữ nguyên tỉ lệ."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
