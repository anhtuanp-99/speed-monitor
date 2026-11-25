"""Ước lượng vận tốc xe theo mô hình mặt đường (xem camera_model.py). Mã thuần, không cần model hay camera.

Cách làm:
  - Chỉ đo xe có điểm bám (giữa cạnh dưới khung xe) nằm trong VÙNG ĐO.
  - Mỗi mẫu (t, u, v) đổi sang vị trí (X, Z) trên mặt đường; vận tốc = độ dốc bền vững (Theil-Sen) của
    (X, Z) theo thời gian, nhân với độ cao camera H và hệ số k:  v = hypot(vx, vz) * H * k * 3.6  (km/h).
  - H lấy từ bề rộng khung xe (trung vị trên nhiều xe) khi bật chế độ tự động, ngược lại dùng H nhập tay.
  - Xe rời vùng đo (hoặc mất ID) thì chốt kết quả cuối cùng và ghi log.
"""
import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from . import config
from .camera_model import CalibParams, CameraModel

LOST_FEW_SAMPLES = "it_mau"          # vào vùng đo nhưng quá ít mẫu / quá ngắn


# ---------------------------------------------------------------------------
# Hàm thuần
# ---------------------------------------------------------------------------
def _robust_slope(ts, values):
    """Độ dốc của values theo ts bằng Theil-Sen (trung vị các độ dốc giữa từng cặp mẫu) để chịu
    được khung nhận diện nhảy bất thường. Chỉ dùng các cặp cách nhau >= 1/3 khoảng thời gian."""
    n = len(ts)
    if n > config.MAX_FIT_POINTS:
        step = n / config.MAX_FIT_POINTS
        keep = sorted({min(n - 1, int(i * step)) for i in range(config.MAX_FIT_POINTS)} | {n - 1})
        ts = [ts[i] for i in keep]
        values = [values[i] for i in keep]
        n = len(ts)
    span = ts[-1] - ts[0]
    if n < 2 or span <= 0:
        return None
    min_dt = span / 3.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = ts[j] - ts[i]
            if dt >= min_dt:
                slopes.append((values[j] - values[i]) / dt)
    return statistics.median(slopes) if slopes else None


def fit_velocity(camera, samples, min_samples, min_span):
    """Vận tốc (vx, vz) và khoảng thời gian của các mẫu (t, u, v, w), tính theo đơn vị H = 1 (1/giây).
    Trả về None nếu không đủ mẫu hợp lệ."""
    pts = []
    for t, u, v, _w in samples:
        g = camera.ground_unit(u, v)
        if g is not None:
            pts.append((t, g[0], g[1]))
    if len(pts) < min_samples:
        return None
    ts = [p[0] for p in pts]
    span = ts[-1] - ts[0]
    if span < min_span:
        return None
    vx = _robust_slope(ts, [p[1] for p in pts])
    vz = _robust_slope(ts, [p[2] for p in pts])
    if vx is None or vz is None:
        return None
    return vx, vz, span


def error_stats(records, true_kmh):
    """So sánh tốc độ đo được của các bản ghi hợp lệ với tốc độ thật. Trả về {"v": {...}} hoặc {"v": None}."""
    errors = [r.speed_kmh - true_kmh for r in records if r.valid]
    if not errors:
        return {"v": None}
    abs_err = [abs(e) for e in errors]
    return {"v": {
        "n": len(errors),
        "bias": statistics.mean(errors),                 # >0: đo cao hơn thật
        "mean_abs": statistics.mean(abs_err),
        "median_abs": statistics.median(abs_err),
        "mean_pct": statistics.mean(abs_err) / true_kmh * 100.0 if true_kmh else float("nan"),
    }}


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
@dataclass
class SpeedRecord:
    track_id: int             # ID hiển thị (đếm từ 1)
    class_name: str
    direction: str            # "ra_xa" | "lai_gan" | "dung"
    t_enter: float            # thời điểm mẫu đầu tiên trong vùng đo (giây)
    t_exit: float             # thời điểm mẫu cuối cùng trong vùng đo (giây)
    speed_kmh: float
    distance_m: float         # quãng đường ước lượng đã đi trong vùng đo
    n_samples: int
    height_m: float           # độ cao camera đã dùng để tính
    auto_height: bool         # True nếu H lấy từ bề rộng xe, False nếu nhập tay
    valid: bool               # tốc độ không vượt MAX_VALID_KMH

    @property
    def transit_s(self):
        return self.t_exit - self.t_enter


@dataclass
class _Track:
    class_name: str
    last_seen: float
    votes: dict = field(default_factory=dict)           # số lần mỗi loại xe được nhận diện
    samples: list = field(default_factory=list)         # (t, u, v, w) trong vùng đo
    live_kmh: float = None
    tried_len: int = 0                                  # số mẫu ở lần chốt thất bại gần nhất
    done: bool = False


class SpeedEstimator:
    def __init__(self, params=None):
        self.params = (params or CalibParams()).clamped()
        self.camera = None
        self._cam_key = None
        self._height_votes = deque(maxlen=config.AUTO_HEIGHT_MAX_TRACKS)
        self.reset()

    def reset(self):
        self._tracks = {}
        self.results = {}          # track_id -> SpeedRecord (để hiển thị nhãn)
        self._height_votes.clear()

    # ---------------- cấu hình ----------------
    def configure(self, width, height, params):
        """Gọi mỗi frame: nhận kích thước ảnh và hệ số hiệu chỉnh hiện tại. Đổi góc nhìn, góc cúi hoặc bề rộng xe
        thì các độ cao H đã tự tính trước đó không còn đúng nên bị xóa."""
        self.params = params.clamped()
        key = (width, height, round(self.params.hfov_deg, 3), round(self.params.pitch_deg, 3),
               tuple(sorted(self.params.widths.items())))
        if key != self._cam_key:
            self._cam_key = key
            self.camera = CameraModel(width, height, self.params.hfov_deg, self.params.pitch_deg)
            self._height_votes.clear()

    def zone_rows(self):
        """(hàng trên, hàng dưới) của vùng đo theo pixel; None nếu vùng rỗng. Cạnh trên bị kẹp để không
        chạm sát đường chân trời."""
        if self.camera is None:
            return None
        h = self.camera.height
        a = h * (1 - self.params.zone_top_pct / 100.0)
        b = h * (1 - self.params.zone_bottom_pct / 100.0)
        top, bottom = min(a, b), max(a, b)
        top = max(top, self.camera.min_row())
        bottom = min(bottom, h)
        return (top, bottom) if top < bottom else None

    def effective_height(self):
        """(H dùng để tính, có phải tự động không, số xe đã dùng để tự tính)."""
        votes = len(self._height_votes)
        if self.params.auto_height and votes >= config.AUTO_HEIGHT_MIN_TRACKS:
            return statistics.median(self._height_votes), True, votes
        return self.params.height_m, False, votes

    def zone_ground_range(self):
        """(khoảng cách gần, khoảng cách xa) của vùng đo tính từ camera, mét; None nếu vùng rỗng."""
        rows = self.zone_rows()
        if rows is None:
            return None
        height = self.effective_height()[0]
        near = self.camera.ground_depth(rows[1], height)
        far = self.camera.ground_depth(rows[0], height)
        return (near, far) if near is not None and far is not None else None

    # ------------------------------------------------------------------
    def update(self, track_id, class_name, t, u, v, box_w):
        """Cập nhật một xe ở thời điểm t (giây). (u, v) là điểm bám (giữa cạnh dưới khung xe), box_w là
        bề rộng khung xe (pixel). Trả về SpeedRecord khi xe vừa rời vùng đo (đủ mẫu), ngược lại None."""
        rows = self.zone_rows()
        if rows is None:
            return None
        track = self._tracks.get(track_id)
        if track is None:
            track = _Track(class_name=class_name, last_seen=t)
            self._tracks[track_id] = track
        track.last_seen = t
        if track.done:
            return None

        # YOLO hay đổi loại xe giữa các frame (car <-> truck...): lấy loại xuất hiện nhiều nhất
        track.votes[class_name] = track.votes.get(class_name, 0) + 1
        track.class_name = max(track.votes, key=track.votes.get)

        if rows[0] <= v <= rows[1]:
            if track.samples and t - track.samples[-1][0] > config.MAX_GAP_S:
                track.samples.clear()                      # mất dấu lâu: bỏ mẫu cũ, đo lại
                track.tried_len = 0
            track.samples.append((t, u, v, box_w))
            if len(track.samples) > config.MAX_SAMPLES_PER_TRACK:
                del track.samples[:len(track.samples) - config.MAX_SAMPLES_PER_TRACK]
            self._update_live(track, t)
            return None

        track.live_kmh = None
        if track.samples:                                  # vừa rời vùng đo
            return self._finish(track_id, track)
        return None

    def _update_live(self, track, now):
        """Tốc độ tức thời trong vùng đo từ các mẫu trong LIVE_WINDOW_S gần nhất, có làm mượt."""
        recent = [s for s in track.samples if now - s[0] <= config.LIVE_WINDOW_S]
        fit = fit_velocity(self.camera, recent, config.LIVE_MIN_SAMPLES, config.LIVE_MIN_SPAN_S)
        value = None
        if fit is not None:
            height = self.effective_height()[0]
            value = math.hypot(fit[0], fit[1]) * height * self.params.k * 3.6
            if value > config.MAX_VALID_KMH * 1.5:
                value = None                               # số phi lý (khung nhảy), không hiện
        if value is None:
            track.live_kmh = None
        elif track.live_kmh is None:
            track.live_kmh = value
        else:
            a = config.LIVE_SMOOTHING
            track.live_kmh = (1 - a) * track.live_kmh + a * value

    def live_speed(self, track_id):
        """Tốc độ tức thời (km/h) của xe đang ở trong vùng đo; None nếu chưa đủ dữ liệu hoặc xe đã chốt."""
        track = self._tracks.get(track_id)
        if track is None or track.done or track.live_kmh is None:
            return None
        return 0.0 if track.live_kmh < config.STOP_KMH else track.live_kmh

    def is_measuring(self, track_id):
        """True nếu xe đang trong vùng đo nhưng chưa đủ mẫu để hiện tốc độ."""
        track = self._tracks.get(track_id)
        return (track is not None and not track.done and bool(track.samples)
                and track.live_kmh is None)

    def _finish(self, track_id, track, force=False):
        if not force and len(track.samples) == track.tried_len:
            return None                                    # chưa có mẫu mới kể từ lần chốt thất bại trước
        fit = fit_velocity(self.camera, track.samples, config.MIN_SAMPLES, config.MIN_SPAN_S)
        track.tried_len = len(track.samples)
        if fit is None:
            return None
        vx, vz, span = fit
        height, auto, _votes = self.effective_height()
        speed_ms = math.hypot(vx, vz) * height * self.params.k
        speed_kmh = speed_ms * 3.6
        if speed_kmh < config.STOP_KMH:
            speed_kmh, speed_ms, direction = 0.0, 0.0, "dung"
        else:
            direction = "ra_xa" if vz > 0 else "lai_gan"

        self._vote_height(track)
        track.done = True
        track.live_kmh = None
        record = SpeedRecord(
            track_id=track_id, class_name=track.class_name, direction=direction,
            t_enter=track.samples[0][0], t_exit=track.samples[-1][0],
            speed_kmh=speed_kmh, distance_m=speed_ms * span, n_samples=len(track.samples),
            height_m=height, auto_height=auto, valid=speed_kmh <= config.MAX_VALID_KMH)
        self.results[track_id] = record
        track.samples = []
        return record

    def _vote_height(self, track):
        """Mỗi xe đã đo xong đóng góp một độ cao H (trung vị các mẫu của xe đó) cho chế độ tự động."""
        width = self.params.widths.get(track.class_name)
        if not width:
            return
        values = [self.camera.height_from_width(w, v, width) for _t, _u, v, w in track.samples]
        values = [x for x in values if x is not None]
        if values:
            self._height_votes.append(statistics.median(values))

    def prune(self, now):
        """Chốt và xóa trạng thái của xe đã vắng mặt quá lâu.
        Trả về (danh sách SpeedRecord chốt muộn, danh sách (id, loại xe, lý do) của xe không đo được)."""
        records, lost = [], []
        for tid, tr in list(self._tracks.items()):
            if now - tr.last_seen <= config.STALE_AFTER_S:
                continue
            if not tr.done and tr.samples:
                record = self._finish(tid, tr, force=True)
                if record is not None:
                    records.append(record)
                else:
                    lost.append((tid, tr.class_name, LOST_FEW_SAMPLES))
            del self._tracks[tid]
            self.results.pop(tid, None)
        return records, lost


class SpeedStats:
    """Thống kê cho giao diện: số xe đã đo, tốc độ theo loại xe, tốc độ dòng xe (công thức 37, 38 của bài báo)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.by_class = {}         # {loại xe: [tốc độ km/h của các xe hợp lệ]}
        self.recent = deque(maxlen=config.RECENT_SHOWN)
        self.invalid = 0
        self.lost = {}             # {loại xe: {lý do: số xe}}

    def add(self, record):
        if not record.valid:
            self.invalid += 1
            return
        self.by_class.setdefault(record.class_name, []).append(record.speed_kmh)
        self.recent.appendleft(record)

    def add_lost(self, class_name, reason):
        reasons = self.lost.setdefault(class_name, {})
        reasons[reason] = reasons.get(reason, 0) + 1

    @property
    def total(self):
        return sum(len(v) for v in self.by_class.values())

    def summary(self):
        """Theo loại xe: số xe, trung bình (công thức 37), nhỏ nhất, lớn nhất."""
        return {name: {"n": len(v), "mean": statistics.mean(v), "min": min(v), "max": max(v)}
                for name, v in self.by_class.items()}

    def flow(self):
        """Tốc độ dòng xe = trung bình có trọng số theo số xe của từng loại (công thức 38) và tình trạng.
        Trả về None nếu chưa có xe."""
        total = self.total
        if total == 0:
            return None
        mean = sum(len(v) * statistics.mean(v) for v in self.by_class.values()) / total
        jam, busy = config.TRAFFIC_THRESHOLDS
        state = "ket" if mean < jam else "dong" if mean < busy else "thong thoang"
        return {"n": total, "mean": mean, "state": state}