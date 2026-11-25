"""Ghi kết quả đo ra file CSV (mỗi phiên một file, nằm trong thư mục của ngày) để theo dõi và tính sai số.

Ví dụ: logs/2026-11-22/speed_20-43-05.csv          (xe đã đo xong)
       logs/2026-11-22/speed_20-43-05_lost.csv     (xe vào vùng đo nhưng không đủ mẫu, tạo khi có xe đầu tiên)
"""
import csv
from datetime import datetime
from pathlib import Path

from . import config

HEADER = ["thoi_diem_s", "id", "loai_xe", "huong", "t_vao_vung_s", "t_ra_vung_s",
          "thoi_gian_trong_vung_s", "quang_duong_m", "toc_do_kmh", "do_cao_cam_m",
          "h_tu_dong", "hop_le"]
LOST_HEADER = ["id", "loai_xe", "ly_do"]


def _unique(path):
    """Hai phiên bắt đầu cùng một giây thì đánh số thêm (_2, _3...) thay vì ghi đè nhau."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


class Recorder:
    def __init__(self, path=None):
        if path is None:
            now = datetime.now()
            path = _unique(config.day_log_dir(now) / f"speed_{now:%H-%M-%S}.csv")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)
        self._file.flush()
        self._lost_file = None
        self._lost_writer = None

    def write(self, record):
        self._writer.writerow([
            f"{record.t_exit:.3f}", record.track_id, record.class_name, record.direction,
            f"{record.t_enter:.3f}", f"{record.t_exit:.3f}", f"{record.transit_s:.3f}",
            f"{record.distance_m:.2f}", f"{record.speed_kmh:.2f}", f"{record.height_m:.2f}",
            int(record.auto_height), int(record.valid)])
        self._file.flush()          # ghi ngay để mở file theo dõi được trong lúc đang chạy

    def write_lost(self, track_id, class_name, reason):
        """Ghi một xe không đo được vào file _lost.csv (tạo file ở lần ghi đầu tiên)."""
        if self._lost_file is None:
            lost_path = self.path.with_name(f"{self.path.stem}_lost{self.path.suffix}")
            self._lost_file = open(lost_path, "w", newline="", encoding="utf-8")
            self._lost_writer = csv.writer(self._lost_file)
            self._lost_writer.writerow(LOST_HEADER)
        self._lost_writer.writerow([track_id, class_name, reason])
        self._lost_file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
        if self._lost_file is not None and not self._lost_file.closed:
            self._lost_file.close()