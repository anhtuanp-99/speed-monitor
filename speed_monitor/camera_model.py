"""Mô hình camera -> mặt đường (hiệu chỉnh ảo, bản rút gọn theo Tran et al. 2026).

Camera lỗ kim đặt ở độ cao H, cúi xuống góc `pitch` so với phương ngang, không xoay (roll = 0).
Với điểm ảnh (u, v) nằm trên mặt đường:
    v' = (v - cy) / f                         (f: tiêu cự pixel, suy ra từ góc nhìn ngang HFOV)
    d  = sin(pitch) + v' * cos(pitch)         (> 0 khi điểm nằm dưới đường chân trời)
    độ sâu theo trục camera  z_c = H / d
    khoảng cách dọc đường    Z   = z_c * (cos(pitch) - v' * sin(pitch))
    độ lệch ngang            X   = (u - cx) / f * z_c
Mọi khoảng cách đều tỉ lệ thuận với H, nên ta tính với H = 1 ("đơn vị") rồi nhân H sau: đổi H hoặc
hệ số k lúc đang chạy không phải tính lại vị trí.

Độ cao H có thể tự suy ra từ bề rộng khung xe: xe loại j có bề rộng thật W_j, bề rộng khung w_px ở hàng v:
    z_c = f * W_j / w_px   =>   H = z_c * d(v)
(trung vị trên nhiều xe, giống ý tưởng dùng kích thước trung vị của loại xe làm "thước" trong bài báo).
Hướng đường (yaw) không ảnh hưởng vì tốc độ là độ dài vector (X, Z).
"""
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import config


def _clamp(value, key):
    low, high = config.LIMITS[key]
    return min(high, max(low, float(value)))


@dataclass
class CalibParams:
    """Các hệ số hiệu chỉnh chỉnh được trên giao diện."""
    hfov_deg: float = config.DEFAULT_HFOV_DEG
    pitch_deg: float = config.DEFAULT_PITCH_DEG
    height_m: float = config.DEFAULT_HEIGHT_M
    auto_height: bool = True
    k: float = config.DEFAULT_K
    zone_top_pct: float = config.DEFAULT_ZONE_TOP_PCT
    zone_bottom_pct: float = config.DEFAULT_ZONE_BOTTOM_PCT
    widths: dict = field(default_factory=lambda: dict(config.VEHICLE_WIDTH_M))

    def copy(self):
        return replace(self, widths=dict(self.widths))

    def clamped(self):
        out = self.copy()
        for key in ("hfov_deg", "pitch_deg", "height_m", "k", "zone_top_pct", "zone_bottom_pct"):
            setattr(out, key, _clamp(getattr(out, key), key))
        out.widths = {str(n): _clamp(w, "width_m") for n, w in out.widths.items()}
        return out

    def to_dict(self):
        return {"hfov_deg": self.hfov_deg, "pitch_deg": self.pitch_deg, "height_m": self.height_m,
                "auto_height": self.auto_height, "k": self.k, "zone_top_pct": self.zone_top_pct,
                "zone_bottom_pct": self.zone_bottom_pct, "widths": dict(self.widths)}

    @classmethod
    def from_dict(cls, data):
        out = cls()
        for key in ("hfov_deg", "pitch_deg", "height_m", "k", "zone_top_pct", "zone_bottom_pct"):
            if key in data:
                try:
                    setattr(out, key, float(data[key]))
                except (TypeError, ValueError):
                    pass
        if "auto_height" in data:
            out.auto_height = bool(data["auto_height"])
        for name, value in (data.get("widths") or {}).items():
            try:
                out.widths[str(name)] = float(value)
            except (TypeError, ValueError):
                pass
        return out.clamped()

    @classmethod
    def load(cls, path=None):
        """Đọc hệ số đã lưu; thiếu file hoặc file hỏng thì dùng mặc định."""
        path = Path(path or config.CALIB_FILE)
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, AttributeError):
            return cls()

    def save(self, path=None):
        path = Path(path or config.CALIB_FILE)
        path.write_text(json.dumps(self.clamped().to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


class CameraModel:
    def __init__(self, width, height, hfov_deg, pitch_deg):
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.f = self.cx / math.tan(math.radians(hfov_deg) / 2.0)
        theta = math.radians(pitch_deg)
        self._sin, self._cos, self._tan = math.sin(theta), math.cos(theta), math.tan(theta)

    # ---------------- đường chân trời & vùng hợp lệ ----------------
    def horizon_row(self):
        """Hàng ảnh của đường chân trời (điểm triệt tiêu của mặt đường). Có thể nằm ngoài khung hình."""
        return self.cy - self.f * self._tan

    def min_row(self):
        """Hàng ảnh nhỏ nhất (xa nhất) còn dùng được: sát chân trời thì hình học phi lý."""
        return self.cy + self.f * (config.MIN_DENOM - self._sin) / self._cos

    def _denominator(self, v):
        return self._sin + (v - self.cy) / self.f * self._cos

    # ---------------- ảnh -> mặt đường ----------------
    def ground_unit(self, u, v):
        """(X, Z) trên mặt đường khi H = 1; None nếu điểm quá sát đường chân trời."""
        d = self._denominator(v)
        if d < config.MIN_DENOM:
            return None
        v_n = (v - self.cy) / self.f
        z_c = 1.0 / d
        return (u - self.cx) / self.f * z_c, (self._cos - v_n * self._sin) * z_c

    def ground_depth(self, v, height_m):
        """Khoảng cách dọc đường (mét) tới hàng ảnh v."""
        g = self.ground_unit(self.cx, v)
        return None if g is None else g[1] * height_m

    def height_from_width(self, w_px, v, real_width_m):
        """Độ cao camera H suy ra từ một xe có bề rộng thật real_width_m, khung rộng w_px, điểm bám ở hàng v."""
        d = self._denominator(v)
        if w_px <= 0 or real_width_m <= 0 or d < config.MIN_DENOM:
            return None
        return self.f * real_width_m / w_px * d

    # ---------------- mặt đường -> ảnh (dùng cho mô phỏng và kiểm thử) ----------------
    def pixel_from_ground(self, x_m, z_m, height_m, lift_m=0.0):
        """Điểm trên đường (X, Z) ở độ cao lift_m so với mặt đường -> (u, v, độ sâu z_c)."""
        y_c = -z_m * self._sin + (height_m - lift_m) * self._cos
        z_c = z_m * self._cos + (height_m - lift_m) * self._sin
        return self.cx + self.f * x_m / z_c, self.cy + self.f * y_c / z_c, z_c