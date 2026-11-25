"""Nhận diện + bám vết: YOLO -> ByteTrack. Chỉ trả về danh sách xe, không biết gì về vạch hay tốc độ."""
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Detection:
    track_id: int
    class_name: str
    cx: float          # tâm khung xe
    cy: float
    w: float           # bề rộng khung xe (pixel)
    h: float

    @property
    def anchor_x(self):
        """Điểm bám = giữa cạnh dưới khung xe (chỗ xe chạm mặt đường)."""
        return self.cx

    @property
    def anchor_y(self):
        return self.cy + self.h / 2.0


class Detector:
    def __init__(self):
        from ultralytics import YOLO      # nạp muộn để phần logic kiểm thử được khi chưa cài ultralytics

        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình: {config.MODEL_PATH}\n"
                "-> Hãy đặt file best.pt vào thư mục weights/")
        self.device = config.DEVICE
        self.device_text = config.describe_device()
        print(f"[Detector] Thiết bị xử lý -> {self.device_text}")
        self.model = YOLO(str(config.MODEL_PATH))

        # Tên lớp lấy từ chính best.pt để không bị lệch thứ tự so với config.CLASS_NAMES
        names = getattr(self.model, "names", None) or config.CLASS_NAMES
        self.names = {int(k): str(v).strip().lower() for k, v in dict(names).items()}
        print("[Detector] Các lớp của model:", self.names)
        missing = sorted({n for n in self.names.values()} - set(config.VEHICLE_WIDTH_M))
        if missing:
            print(f"[Detector] CẢNH BÁO: lớp {missing} không có trong config.VEHICLE_WIDTH_M "
                  "-> Mức 2 sẽ không tính được cho các lớp này")

        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                           device=self.device, verbose=False)
        self.reset()

    def reset(self):
        """Xóa trạng thái bám vết (gọi khi bắt đầu phiên mới hoặc khi video phát lại)."""
        self._fresh_tracker = True

    def detect(self, frame):
        """Trả về danh sách Detection của frame hiện tại."""
        results = self.model.track(
            frame, persist=not self._fresh_tracker, tracker=config.TRACKER_TYPE,
            conf=config.DETECT_CONF, agnostic_nms=config.AGNOSTIC_NMS,
            device=self.device, verbose=False)
        self._fresh_tracker = False

        boxes = results[0].boxes
        if boxes.id is None:
            return []
        xywh = boxes.xywh.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        return [Detection(tid, self.names.get(cid, "unknown"),
                          float(cx), float(cy), float(w), float(h))
                for (cx, cy, w, h), tid, cid in zip(xywh, track_ids, class_ids)]