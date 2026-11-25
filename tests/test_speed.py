import tempfile
import unittest
from pathlib import Path

from speed_monitor import config
from speed_monitor.camera_model import CalibParams, CameraModel
from speed_monitor.speed_estimator import SpeedRecord, SpeedStats, error_stats
from tests.simulation import FRAME_H, FRAME_W, Scene, drive, make_estimator


class CameraGeometry(unittest.TestCase):
    def test_ground_roundtrip(self):
        cam = CameraModel(FRAME_W, FRAME_H, 65, 10)
        u, v, _ = cam.pixel_from_ground(2.0, 30.0, 7.0)
        x1, z1 = cam.ground_unit(u, v)
        self.assertAlmostEqual(x1 * 7.0, 2.0, places=6)
        self.assertAlmostEqual(z1 * 7.0, 30.0, places=6)

    def test_horizon_row_matches_pitch(self):
        cam = CameraModel(FRAME_W, FRAME_H, 65, 0)
        self.assertAlmostEqual(cam.horizon_row(), FRAME_H / 2)
        self.assertLess(CameraModel(FRAME_W, FRAME_H, 65, 20).horizon_row(), FRAME_H / 2)

    def test_height_from_width(self):
        cam = CameraModel(FRAME_W, FRAME_H, 65, 10)
        u, v, _ = cam.pixel_from_ground(0.0, 25.0, 7.0)
        _, _, depth = cam.pixel_from_ground(0.0, 25.0, 7.0)
        w_px = cam.f * 1.74 / depth
        self.assertAlmostEqual(cam.height_from_width(w_px, v, 1.74), 7.0, places=6)

    def test_points_above_horizon_rejected(self):
        cam = CameraModel(FRAME_W, FRAME_H, 65, 10)
        self.assertIsNone(cam.ground_unit(480, cam.horizon_row() - 5))


class Estimator(unittest.TestCase):
    def test_manual_height_exact(self):
        scene = Scene()
        est = make_estimator(auto_height=False, height_m=7.0)
        recs = drive(est, scene, 1, "car", 40.0)
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].speed_kmh, 40.0, delta=0.4)
        self.assertEqual(recs[0].direction, "lai_gan")
        self.assertFalse(recs[0].auto_height)

    def test_direction_away(self):
        scene = Scene()
        est = make_estimator(auto_height=False, height_m=7.0)
        recs = drive(est, scene, 1, "bus", 30.0, toward=False)
        self.assertEqual(recs[0].direction, "ra_xa")
        self.assertAlmostEqual(recs[0].speed_kmh, 30.0, delta=0.4)

    def test_speed_independent_of_lane_and_distance(self):
        scene = Scene()
        for x_m in (-4.0, 0.0, 5.0):
            est = make_estimator(auto_height=False, height_m=7.0)
            rec = drive(est, scene, 1, "car", 50.0, x_m=x_m)[0]
            self.assertAlmostEqual(rec.speed_kmh, 50.0, delta=0.6)

    def test_k_scales_speed(self):
        scene = Scene()
        est = make_estimator(auto_height=False, height_m=7.0, k=1.1)
        rec = drive(est, scene, 1, "car", 40.0)[0]
        self.assertAlmostEqual(rec.speed_kmh, 44.0, delta=0.5)

    def test_auto_height_recovers_true_height(self):
        scene = Scene(height_m=9.0)
        est = make_estimator(auto_height=True, height_m=3.0)      # H nhập tay cố tình sai
        rng_speeds = [30, 35, 40, 45, 50, 38, 42, 33, 47, 36]
        classes = ["car", "motorbike", "car", "truck", "motorbike", "bus", "car", "motorbike", "car", "truck"]
        recs = []
        for i, (kmh, name) in enumerate(zip(rng_speeds, classes), start=1):
            recs += drive(est, scene, i, name, kmh, x_m=(i % 3 - 1) * 3.0, t0=i * 20.0)
        height, auto, votes = est.effective_height()
        self.assertTrue(auto)
        self.assertEqual(votes, 10)
        self.assertAlmostEqual(height, 9.0, delta=0.4)            # lệch < 5% (do tâm mặt sau xe cao hơn mặt đường)
        late = recs[-1]
        self.assertTrue(late.auto_height)
        self.assertAlmostEqual(late.speed_kmh, rng_speeds[-1], delta=0.06 * rng_speeds[-1])

    def test_stationary_vehicle_reported_as_stopped(self):
        scene = Scene()
        est = make_estimator(auto_height=False, height_m=7.0)
        recs = drive(est, scene, 1, "car", 0.5)               # đi rất chậm: rời vùng đo sau vài trăm giây
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].speed_kmh, 0.0)
        self.assertEqual(recs[0].direction, "dung")

    def test_samples_per_track_are_capped(self):
        est = make_estimator(auto_height=False)
        top, bottom = est.zone_rows()
        mid = (top + bottom) / 2
        for k in range(config.MAX_SAMPLES_PER_TRACK + 500):
            est.update(1, "car", k / 30.0, 480, mid, 50)      # xe đứng yên trong vùng đo
        self.assertEqual(len(est._tracks[1].samples), config.MAX_SAMPLES_PER_TRACK)

    def test_too_few_samples_is_lost(self):
        est = make_estimator(auto_height=False)
        top, bottom = est.zone_rows()
        mid = (top + bottom) / 2
        for k in range(3):
            est.update(1, "car", k / 30.0, 480, mid + k, 50)
        records, lost = est.prune(100.0)
        self.assertEqual(records, [])
        self.assertEqual(lost, [(1, "car", "it_mau")])

    def test_vehicle_outside_zone_ignored(self):
        est = make_estimator(auto_height=False)
        for k in range(30):
            self.assertIsNone(est.update(1, "car", k / 30.0, 480, 5, 50))      # sát mép trên, trên chân trời
        self.assertEqual(est.prune(100.0), ([], []))

    def test_zone_clamped_below_horizon(self):
        est = make_estimator(zone_top_pct=100.0, zone_bottom_pct=0.0)
        top, _bottom = est.zone_rows()
        self.assertGreater(top, est.camera.horizon_row())

    def test_class_majority_vote(self):
        est = make_estimator(auto_height=False)
        top, bottom = est.zone_rows()
        mid = (top + bottom) / 2
        for k, name in enumerate(["car", "truck", "car", "car"]):
            est.update(1, name, k / 30.0, 480, mid, 50)
        self.assertEqual(est._tracks[1].class_name, "car")

    def test_noise_robustness(self):
        scene = Scene()
        noise = {"u": 1.0, "v": 1.5, "w": 0.03}
        est = make_estimator(auto_height=False, height_m=7.0)
        rec = drive(est, scene, 1, "car", 40.0, noise=noise)[0]
        self.assertAlmostEqual(rec.speed_kmh, 40.0, delta=3.0)


class Params(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            p = CalibParams(hfov_deg=70, pitch_deg=12.5, k=1.07, auto_height=False)
            p.widths["motorbike"] = 0.8
            p.save(path)
            q = CalibParams.load(path)
            self.assertEqual((q.hfov_deg, q.pitch_deg, q.k, q.auto_height), (70, 12.5, 1.07, False))
            self.assertEqual(q.widths["motorbike"], 0.8)

    def test_missing_or_corrupt_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(CalibParams.load(Path(tmp) / "none.json").k, config.DEFAULT_K)
            bad = Path(tmp) / "bad.json"
            bad.write_text("{không phải json", encoding="utf-8")
            self.assertEqual(CalibParams.load(bad).pitch_deg, config.DEFAULT_PITCH_DEG)

    def test_values_are_clamped(self):
        q = CalibParams.from_dict({"k": 99, "pitch_deg": -5, "widths": {"car": 50}})
        self.assertEqual(q.k, config.LIMITS["k"][1])
        self.assertEqual(q.pitch_deg, config.LIMITS["pitch_deg"][0])
        self.assertEqual(q.widths["car"], config.LIMITS["width_m"][1])


class Stats(unittest.TestCase):
    def _rec(self, i, name, kmh, valid=True):
        return SpeedRecord(i, name, "lai_gan", 0.0, 1.0, kmh, 5.0, 20, 7.0, True, valid)

    def test_summary_and_flow(self):
        st = SpeedStats()
        st.add(self._rec(1, "car", 30.0))
        st.add(self._rec(2, "car", 50.0))
        st.add(self._rec(3, "motorbike", 20.0))
        st.add(self._rec(4, "bus", 300.0, valid=False))
        st.add_lost("car", "it_mau")
        self.assertEqual(st.total, 3)
        self.assertEqual(st.invalid, 1)
        car = st.summary()["car"]
        self.assertEqual((car["n"], car["mean"], car["min"], car["max"]), (2, 40.0, 30.0, 50.0))
        flow = st.flow()                           # (2*40 + 1*20) / 3 = 33.3 (công thức 38)
        self.assertAlmostEqual(flow["mean"], 100 / 3, places=6)
        self.assertEqual(flow["state"], "thong thoang")
        self.assertEqual(st.lost, {"car": {"it_mau": 1}})

    def test_traffic_states(self):
        for kmh, state in ((10.0, "ket"), (20.0, "dong"), (45.0, "thong thoang")):
            st = SpeedStats()
            st.add(self._rec(1, "car", kmh))
            self.assertEqual(st.flow()["state"], state)
        self.assertIsNone(SpeedStats().flow())

    def test_error_stats(self):
        recs = [self._rec(1, "car", 38.0), self._rec(2, "car", 42.0), self._rec(3, "car", 99.0, valid=False)]
        s = error_stats(recs, 40.0)["v"]
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["bias"], 0.0)
        self.assertAlmostEqual(s["mean_abs"], 2.0)
        self.assertIsNone(error_stats([], 40.0)["v"])


if __name__ == "__main__":
    unittest.main()