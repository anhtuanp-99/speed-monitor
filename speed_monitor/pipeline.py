"""Nối các bước cho một frame: nhận diện -> ước lượng tốc độ -> vẽ -> ghi CSV.

Giao diện (app.py) và dòng lệnh (evaluate.py) cùng gọi lớp này nên logic chỉ có một bản.
ID hiển thị đếm tuần tự 1, 2, 3, ... theo thứ tự xe xuất hiện (không dùng ID thô của ByteTrack, vốn nhảy cóc).
"""
from dataclasses import replace

from . import drawing
from .camera_model import CalibParams
from .speed_estimator import SpeedEstimator, SpeedStats


class SpeedPipeline:
    def __init__(self, detector=None, recorder=None, params=None):
        if detector is None:
            from .processor import Detector
            detector = Detector()
        self.detector = detector
        self.recorder = recorder
        self.params = (params or CalibParams()).clamped()   # giao diện / dòng lệnh gán lại mỗi khi chỉnh hệ số
        self.estimator = SpeedEstimator(self.params)
        self.stats = SpeedStats()
        self.records = []              # mọi bản ghi của phiên (dùng cho evaluate.py)
        self.lost_records = []         # xe vào vùng đo nhưng không đủ mẫu: (id, loại xe, lý do)
        self.active_vehicles = 0
        self._display_ids = {}         # ID thô của tracker -> ID hiển thị
        self._next_id = 1

    def reset(self):
        """Bắt đầu phiên mới (xóa bám vết, số liệu, danh sách bản ghi, đánh số lại từ 1)."""
        self.detector.reset()
        self.estimator.reset()
        self.stats.reset()
        self.records = []
        self.lost_records = []
        self._display_ids = {}
        self._next_id = 1

    def _display_id(self, raw_id):
        if raw_id not in self._display_ids:
            self._display_ids[raw_id] = self._next_id
            self._next_id += 1
        return self._display_ids[raw_id]

    def info(self):
        """Thông tin hiệu chỉnh đang dùng, để giao diện hiển thị."""
        height, auto, votes = self.estimator.effective_height()
        camera = self.estimator.camera
        return {"height_m": height, "auto": auto, "votes": votes,
                "zone_range": self.estimator.zone_ground_range(),
                "horizon_row": camera.horizon_row() if camera else None,
                "frame_h": camera.height if camera else None}

    def _accept(self, record):
        self.records.append(record)
        self.stats.add(record)
        if self.recorder is not None:
            self.recorder.write(record)

    def process(self, frame, t):
        """Xử lý một frame ở thời điểm t (giây). Trả về (ảnh đã vẽ, danh sách bản ghi mới)."""
        height, width = frame.shape[:2]
        self.estimator.configure(width, height, self.params)

        detections = self.detector.detect(frame)
        self.active_vehicles = len(detections)

        new_records = []
        for raw in detections:
            det = replace(raw, track_id=self._display_id(raw.track_id))
            record = self.estimator.update(det.track_id, det.class_name, t,
                                           det.anchor_x, det.anchor_y, det.w)
            if record is not None:
                new_records.append(record)
                self._accept(record)
            drawing.draw_vehicle(frame, det, self.estimator.results.get(det.track_id),
                                 self.estimator.live_speed(det.track_id),
                                 self.estimator.is_measuring(det.track_id))

        info = self.info()
        drawing.draw_zone(frame, self.estimator.zone_rows(), info["horizon_row"],
                          self.params.pitch_deg, info["zone_range"])
        drawing.draw_info(frame, self.stats.total, info["height_m"], info["auto"], info["votes"])

        late_records, lost = self.estimator.prune(t)
        for record in late_records:
            new_records.append(record)
            self._accept(record)
        for track_id, class_name, reason in lost:
            self.lost_records.append((track_id, class_name, reason))
            self.stats.add_lost(class_name, reason)
            if self.recorder is not None:
                self.recorder.write_lost(track_id, class_name, reason)
        return frame, new_records