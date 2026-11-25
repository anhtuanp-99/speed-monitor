"""Mô phỏng camera + xe chạy thẳng đều để kiểm tra thuật toán với đáp án biết trước (không cần video thật)."""
import random

from speed_monitor import config
from speed_monitor.camera_model import CalibParams, CameraModel
from speed_monitor.speed_estimator import SpeedEstimator

FPS = 30.0
FRAME_W, FRAME_H = 960, 540
HALF_HEIGHT_M = {"motorbike": 0.8, "car": 0.75, "bus": 1.6, "truck": 1.2}   # độ cao tâm mặt sau xe


class Scene:
    """Camera thật: độ cao, góc cúi, góc nhìn. Các hệ số ước lượng có thể khác để xem độ nhạy."""

    def __init__(self, height_m=7.0, pitch_deg=10.0, hfov_deg=65.0):
        self.height_m, self.pitch_deg, self.hfov_deg = height_m, pitch_deg, hfov_deg
        self.camera = CameraModel(FRAME_W, FRAME_H, hfov_deg, pitch_deg)


def drive(est, scene, track_id, class_name, speed_kmh, x_m=0.0, toward=True, true_width=None,
          z_start=60.0, z_end=5.0, noise=None, rng=None, t0=0.0):
    """Cho một xe chạy thẳng đều qua cảnh, đưa từng frame vào estimator. Trả về danh sách bản ghi sinh ra."""
    rng = rng or random.Random(1)
    noise = noise or {}
    width = true_width if true_width is not None else config.VEHICLE_WIDTH_M[class_name]
    speed_ms = speed_kmh / 3.6
    zs = []
    z = z_start if toward else z_end
    while (z >= z_end if toward else z <= z_start):
        zs.append(z)
        z += (-speed_ms if toward else speed_ms) / FPS
    records = []
    for k, z in enumerate(zs):
        u, v, _zc = scene.camera.pixel_from_ground(x_m, z, scene.height_m)
        if not (0 <= v < FRAME_H and 0 <= u < FRAME_W):
            continue
        _u, _v, depth = scene.camera.pixel_from_ground(x_m, z, scene.height_m, HALF_HEIGHT_M[class_name])
        w_px = scene.camera.f * width / depth
        u += rng.gauss(0, noise.get("u", 0.0))
        v += rng.gauss(0, noise.get("v", 0.0))
        w_px *= 1.0 + noise.get("w_bias", 0.0) + rng.gauss(0, noise.get("w", 0.0))
        record = est.update(track_id, class_name, t0 + k / FPS, u, v, w_px)
        if record is not None:
            records.append(record)
    return records


def make_estimator(**overrides):
    params = CalibParams()
    for key, value in overrides.items():
        if key == "widths":
            params.widths.update(value)
        else:
            setattr(params, key, value)
    est = SpeedEstimator(params)
    est.configure(FRAME_W, FRAME_H, params)
    return est