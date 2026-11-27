import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speed_monitor.camera_model import CalibParams
from speed_monitor.pipeline import SpeedPipeline
from speed_monitor.session_log import SessionLog, build_summary, diff_params
from tests.simulation import FPS, FRAME_H, FRAME_W, Scene
from tests.test_pipeline import FakeDetector, merge, track_script


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DiffTest(unittest.TestCase):
    def test_scalar_and_width_changes(self):
        a = CalibParams()
        b = CalibParams(k=1.07, pitch_deg=12.0)
        b.widths["motorbike"] = 0.8
        d = diff_params(a, b)
        self.assertEqual(set(d), {"k", "pitch_deg", "be_rong_motorbike_m"})
        self.assertEqual(d["k"], {"tu": 1.0, "sang": 1.07})
        self.assertEqual(d["be_rong_motorbike_m"]["sang"], 0.8)

    def test_no_change_is_empty(self):
        self.assertEqual(diff_params(CalibParams(), CalibParams()), {})


class SessionLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "speed_x.json"

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, **kw):
        return SessionLog(self.path, source="test1.mp4", csv_name="speed_x.csv", preset_name="mac_dinh",
                          params=CalibParams(), **kw)

    def test_initial_file_written_and_valid(self):
        self.make()
        doc = read(self.path)
        self.assertEqual(doc["phien"]["nguon_video"], "test1.mp4")
        self.assertEqual(doc["phien"]["trang_thai"], "dang_chay")
        self.assertIsNone(doc["phien"]["ket_thuc"])
        self.assertEqual(doc["he_so_ban_dau"]["k"], 1.0)
        self.assertEqual(doc["thay_doi_thong_so"], [])
        self.assertIsNone(doc["ket_qua"])

    def test_no_change_no_entries(self):
        log = self.make()
        for i in range(100):
            self.assertFalse(log.observe(CalibParams(), "mac_dinh", i / FPS, now=i * 0.033))
        self.assertEqual(read(self.path)["thay_doi_thong_so"], [])

    def test_change_logged_only_after_settling(self):
        log = self.make()
        new = CalibParams(k=1.05)
        self.assertFalse(log.observe(new, "mac_dinh", 10.0, now=100.0))
        self.assertFalse(log.observe(new, "mac_dinh", 10.5, now=100.5))      # chưa đủ 1 giây
        self.assertEqual(read(self.path)["thay_doi_thong_so"], [])
        self.assertTrue(log.observe(new, "mac_dinh", 11.1, now=101.1))
        entries = read(self.path)["thay_doi_thong_so"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["thay_doi"], {"k": {"tu": 1.0, "sang": 1.05}})
        self.assertEqual(entries[0]["thoi_diem_video_s"], 11.1)

    def test_dragging_a_slider_makes_a_single_entry(self):
        log = self.make()
        now = 50.0
        for step in range(1, 101):                              # kéo liên tục 100 giá trị, mỗi 0,03 giây
            log.observe(CalibParams(k=1.0 + step * 0.001), "mac_dinh", 5.0, now=now)
            now += 0.03
        for _ in range(40):                                     # rồi thả tay, đứng yên
            log.observe(CalibParams(k=1.1), "mac_dinh", 6.0, now=now)
            now += 0.05
        entries = read(self.path)["thay_doi_thong_so"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["thay_doi"]["k"]["sang"], 1.1)

    def test_change_reverted_before_settling_is_ignored(self):
        log = self.make()
        log.observe(CalibParams(k=1.3), "mac_dinh", 1.0, now=10.0)
        log.observe(CalibParams(), "mac_dinh", 1.2, now=10.3)    # quay về giá trị cũ
        log.observe(CalibParams(), "mac_dinh", 3.0, now=12.0)
        self.assertEqual(read(self.path)["thay_doi_thong_so"], [])

    def test_two_separate_changes_make_two_entries(self):
        log = self.make()
        for now, video_t, k in ((10.0, 1.0, 1.1), (11.5, 2.5, 1.1), (20.0, 9.0, 1.2), (21.5, 10.5, 1.2)):
            log.observe(CalibParams(k=k), "mac_dinh", video_t, now=now)
        entries = read(self.path)["thay_doi_thong_so"]
        self.assertEqual([e["thay_doi"]["k"] for e in entries],
                         [{"tu": 1.0, "sang": 1.1}, {"tu": 1.1, "sang": 1.2}])

    def test_preset_switch_is_logged(self):
        log = self.make()
        loaded = CalibParams(pitch_deg=9.0)
        log.observe(loaded, "Cầu rộng", 4.0, now=1.0)
        log.observe(loaded, "Cầu rộng", 5.5, now=2.5)
        entry = read(self.path)["thay_doi_thong_so"][0]
        self.assertEqual(entry["bo_he_so"], {"tu": "mac_dinh", "sang": "Cầu rộng"})
        self.assertEqual(entry["thay_doi"]["pitch_deg"], {"tu": 10.0, "sang": 9.0})
        self.assertEqual(read(self.path)["phien"]["bo_he_so"], "Cầu rộng")

    def test_finish_flushes_pending_and_writes_summary(self):
        log = self.make()
        log.observe(CalibParams(k=1.2), "mac_dinh", 7.0, now=1.0)       # đang đổi thì phiên kết thúc
        log.finish({"so_xe_hop_le": 3}, "dung", video_t=7.4)
        doc = read(self.path)
        self.assertEqual(doc["phien"]["trang_thai"], "dung")
        self.assertIsNotNone(doc["phien"]["ket_thuc"])
        self.assertEqual(doc["ket_qua"], {"so_xe_hop_le": 3})
        self.assertEqual(len(doc["thay_doi_thong_so"]), 1)

    def test_no_temp_file_left_behind(self):
        log = self.make()
        log.finish({}, "het_video")
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ["speed_x.json"])

    def test_extra_fields_in_session(self):
        SessionLog(self.path, source="a.mp4", extra={"lenh": ["--source", "a.mp4"]})
        self.assertEqual(read(self.path)["phien"]["lenh"], ["--source", "a.mp4"])


class SummaryTest(unittest.TestCase):
    def run_pipeline(self):
        scene = Scene()
        script = merge(track_script(scene, 1, "car", 30.0),
                       track_script(scene, 2, "motorbike", 50.0, t_start_frame=10, x_m=3.0))
        pipe = SpeedPipeline(detector=FakeDetector(script),
                             params=CalibParams(auto_height=False, height_m=7.0))
        pipe.reset()
        for k in range(400):
            pipe.process(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8), k / FPS)
        return pipe

    def test_summary_contents_and_error(self):
        summary = build_summary(self.run_pipeline(), true_speed_kmh=40.0)
        self.assertEqual(summary["so_xe_hop_le"], 2)
        self.assertEqual(set(summary["theo_loai_xe"]), {"car", "motorbike"})
        self.assertAlmostEqual(summary["theo_loai_xe"]["car"]["tb_kmh"], 30.0, delta=0.8)
        self.assertAlmostEqual(summary["dong_xe"]["tb_kmh"], 40.0, delta=1.0)
        self.assertEqual(summary["dong_xe"]["tinh_trang"], "thong thoang")
        self.assertFalse(summary["do_cao_camera"]["tu_dong"])
        self.assertEqual(summary["sai_so_so_voi_toc_do_that"]["so_xe"], 2)
        self.assertEqual(len(summary["vung_do_cach_camera_m"]), 2)
        json.dumps(summary, ensure_ascii=False)                         # phải ghi được ra JSON

    def test_no_true_speed_means_no_error_section(self):
        self.assertNotIn("sai_so_so_voi_toc_do_that", build_summary(self.run_pipeline()))


if __name__ == "__main__":
    unittest.main()