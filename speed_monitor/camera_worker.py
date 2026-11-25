"""Luồng nền: đọc video -> pipeline -> cập nhật số liệu và ảnh cho giao diện."""
import threading
import time
import traceback
from dataclasses import dataclass, field

import cv2

from . import config
from .camera_model import CalibParams
from .pipeline import SpeedPipeline
from .recorder import Recorder
from .video_source import (VideoStream, fit_to_box, resize_to_width, resolve_source)


@dataclass
class WorkerStatus:
    state: str = "idle"          # idle | loading | running | stopped | error
    message: str = ""
    measured: int = 0
    invalid: int = 0
    active: int = 0
    fps: float = 0.0
    summary: dict = field(default_factory=dict)     # {loại xe: {n, mean, min, max}}
    recent: list = field(default_factory=list)      # các SpeedRecord gần nhất
    lost: dict = field(default_factory=dict)        # {loại xe: {lý do: số xe}} chưa đo xong
    flow: dict = None                               # {n, mean, state}: tốc độ dòng xe
    info: dict = field(default_factory=dict)        # H đang dùng, vùng đo, đường chân trời
    csv_path: str = ""


class CameraWorker:
    def __init__(self, params=None):
        self.pipeline = None
        self.status = WorkerStatus()
        self.params = (params or CalibParams()).clamped()    # hệ số hiệu chỉnh, giao diện cập nhật bất cứ lúc nào
        self.device_text = ""         # GPU hay CPU đang dùng (hiện trên giao diện)

        self.latest_frame = None      # ảnh RGB để giao diện hiển thị
        self.display_size = (config.VIDEO_PLACEHOLDER_W, config.VIDEO_PLACEHOLDER_H)
        self.frame_id = 0
        self._thread = None
        self._stop = threading.Event()
        self._recorder = None

    # ---------------- điều khiển ----------------
    @property
    def is_active(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_stopping(self):
        return self.is_active and self._stop.is_set()

    def start(self, source_text):
        if self.is_active:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source_text,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=3.0):
        if self._thread is not None:
            self._thread.join(timeout)

    def set_params(self, params):
        self.params = params.clamped()

    def set_display_size(self, width, height):
        self.display_size = (max(160, int(width)), max(90, int(height)))

    # ---------------- luồng chính ----------------
    def _publish(self, state, message="", fps=0.0):
        stats = self.pipeline.stats if self.pipeline else None
        self.status = WorkerStatus(
            state=state, message=message, fps=round(fps, 1),
            measured=stats.total if stats else 0,
            invalid=stats.invalid if stats else 0,
            active=self.pipeline.active_vehicles if self.pipeline else 0,
            summary=stats.summary() if stats else {},
            recent=list(stats.recent) if stats else [],
            lost={k: dict(v) for k, v in stats.lost.items()} if stats else {},
            flow=stats.flow() if stats else None,
            info=self.pipeline.info() if self.pipeline else {},
            csv_path=str(self._recorder.path) if self._recorder else "")

    def _publish_frame(self, frame):
        disp = fit_to_box(frame, *self.display_size)
        self.latest_frame = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.frame_id += 1

    @staticmethod
    def _sync_playback(stream, t0):
        """Phát file đúng tốc độ gốc: chạy nhanh thì chờ; xử lý chậm thì bỏ bớt frame
        (tối đa config.MAX_FRAME_SKIP, mặc định 0 = không bỏ frame nào để tracker không mất ID).
        Mốc thời gian tính theo số frame đã đọc nên tốc độ đo vẫn đúng."""
        target = (time.monotonic() - t0) * stream.fps
        if stream.frames_read > target:
            time.sleep(min((stream.frames_read - target) / stream.fps, 0.1))
        elif config.MAX_FRAME_SKIP > 0:
            stream.skip(min(int(target - stream.frames_read), config.MAX_FRAME_SKIP))

    def _run(self, source_text):
        stream = None
        self._recorder = None
        try:
            self._publish("loading", "Đang nạp mô hình...")
            if self.pipeline is None:
                self.pipeline = SpeedPipeline()
            self.device_text = getattr(self.pipeline.detector, "device_text", "")
            self._recorder = Recorder()
            self.pipeline.recorder = self._recorder
            self.pipeline.params = self.params
            self.pipeline.reset()

            stream = VideoStream(resolve_source(source_text))
            t0, fps_ema, last = time.monotonic(), 0.0, time.monotonic()

            while not self._stop.is_set():
                if stream.is_file:
                    self._sync_playback(stream, t0)

                ok, frame, t = stream.read()
                if not ok:
                    if stream.is_file and config.LOOP_VIDEO_FILES:
                        stream.restart()
                        self.pipeline.reset()
                        t0 = time.monotonic()
                        continue
                    break

                frame = resize_to_width(frame, config.PROCESS_WIDTH)
                self.pipeline.params = self.params
                annotated, _ = self.pipeline.process(frame, t)

                now = time.monotonic()
                dt = now - last
                last = now
                if dt > 0:
                    fps_ema = (0.9 * fps_ema + 0.1 / dt) if fps_ema else 1 / dt

                self._publish_frame(annotated)
                self._publish("running", fps=fps_ema)

            self._publish("stopped", "Đã dừng")
        except Exception as exc:                       # hiển thị lỗi lên giao diện
            traceback.print_exc()
            self._publish("error", str(exc))
        finally:
            if stream is not None:
                stream.release()
            if self._recorder is not None:
                self._recorder.close()
            if self.pipeline is not None:
                self.pipeline.recorder = None