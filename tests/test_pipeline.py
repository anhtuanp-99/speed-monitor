import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speed_monitor.camera_model import CalibParams
from speed_monitor.pipeline import SpeedPipeline
from speed_monitor.processor import Detection
from speed_monitor.recorder import Recorder
from tests.simulation import FPS, FRAME_H, FRAME_W, Scene, make_estimator


def track_script(scene, raw_id, class_name, speed_kmh, t_start_frame=0, x_m=0.0, z_start=60.0,
                 z_stop=5.0, length=400):
    """Danh sách frame (mỗi frame là danh sách Detection) của một xe chạy lại gần camera."""
    frames = [[] for _ in range(length)]
    z, k = z_start, 0
    while z >= z_stop and t_start_frame + k < length:
        u, v, depth = scene.camera.pixel_from_ground(x_m, z, scene.height_m, 0.75)
        _u, v0, _ = scene.camera.pixel_from_ground(x_m, z, scene.height_m)
        w_px = scene.camera.f * 1.74 / depth
        if 0 <= v0 < FRAME_H and 0 <= u < FRAME_W:
            frames[t_start_frame + k].append(Detection(raw_id, class_name, u, v0 - 20.0, w_px, 40.0))
        z -= speed_kmh / 3.6 / FPS
        k += 1
    return frames


def merge(*scripts):
    out = [[] for _ in range(max(len(s) for s in scripts))]
    for s in scripts:
        for i, dets in enumerate(s):
            out[i].extend(dets)
    return out


class FakeDetector:
    def __init__(self, script):
        self.script, self.i = script, 0

    def reset(self):
        self.i = 0

    def detect(self, frame):
        dets = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return dets


def run(pipe, n_frames):
    records = []
    for k in range(n_frames):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        _, recs = pipe.process(frame, k / FPS)
        records.extend(recs)
    return records


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = Recorder(Path(self.tmp.name) / "s.csv")
        self.scene = Scene()
        self.params = CalibParams(auto_height=False, height_m=7.0)

    def tearDown(self):
        self.rec.close()
        self.tmp.cleanup()

    def make(self, script):
        pipe = SpeedPipeline(detector=FakeDetector(script), recorder=self.rec, params=self.params)
        pipe.reset()
        return pipe

    def test_speed_display_id_and_csv(self):
        pipe = self.make(track_script(self.scene, 57, "car", 40.0))
        recs = run(pipe, 400)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].track_id, 1)                  # ID hiển thị đếm từ 1, không phải 57
        self.assertAlmostEqual(recs[0].speed_kmh, 40.0, delta=0.6)
        self.rec.close()
        with open(self.rec.path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][:4], ["thoi_diem_s", "id", "loai_xe", "huong"])
        self.assertIn("toc_do_kmh", rows[0])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1:4], ["1", "car", "lai_gan"])
        self.assertAlmostEqual(float(rows[1][rows[0].index("toc_do_kmh")]), 40.0, delta=0.6)

    def test_display_ids_are_sequential(self):
        script = merge(track_script(self.scene, 900, "motorbike", 35.0),
                       track_script(self.scene, 31, "car", 45.0, t_start_frame=20, x_m=3.0))
        pipe = self.make(script)
        recs = run(pipe, 400)
        self.assertEqual(sorted(r.track_id for r in recs), [1, 2])
        by_id = {r.track_id: r for r in recs}
        self.assertEqual(by_id[1].class_name, "motorbike")
        self.assertAlmostEqual(by_id[2].speed_kmh, 45.0, delta=0.8)

    def test_flow_statistics(self):
        script = merge(track_script(self.scene, 1, "car", 30.0),
                       track_script(self.scene, 2, "car", 50.0, t_start_frame=10, x_m=3.0))
        pipe = self.make(script)
        run(pipe, 400)
        self.assertAlmostEqual(pipe.stats.flow()["mean"], 40.0, delta=1.0)

    def test_lost_vehicle_logged(self):
        # Xe vào vùng đo rồi biến mất sau 3 frame
        script = track_script(self.scene, 5, "car", 40.0)
        top, bottom = make_estimator(auto_height=False, height_m=7.0).zone_rows()
        in_zone = [i for i, f in enumerate(script) if f and top <= f[0].anchor_y <= bottom]
        keep = set(in_zone[:3])
        script = [f if i in keep else [] for i, f in enumerate(script)]
        pipe = self.make(script)
        run(pipe, 400)
        self.assertEqual(pipe.lost_records, [(1, "car", "it_mau")])
        self.assertEqual(pipe.stats.lost, {"car": {"it_mau": 1}})
        self.rec.close()
        lost_path = self.rec.path.with_name("s_lost.csv")
        self.assertTrue(lost_path.exists())

    def test_reset_restarts_ids(self):
        script = track_script(self.scene, 7, "car", 40.0)
        pipe = self.make(script)
        run(pipe, 400)
        pipe.reset()
        pipe.detector.script = track_script(self.scene, 7, "car", 40.0)
        recs = run(pipe, 400)
        self.assertEqual(recs[0].track_id, 1)

    def test_params_change_applies_next_frame(self):
        pipe = self.make(track_script(self.scene, 1, "car", 40.0))
        pipe.params = CalibParams(auto_height=False, height_m=7.0, k=1.1)
        recs = run(pipe, 400)
        self.assertAlmostEqual(recs[0].speed_kmh, 44.0, delta=0.8)


if __name__ == "__main__":
    unittest.main()