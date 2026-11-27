import tempfile
import unittest
from pathlib import Path

from speed_monitor.camera_model import CalibParams
from speed_monitor.presets import PresetStore, safe_name, same_video_warning, source_key


class SafeNames(unittest.TestCase):
    def test_removes_forbidden_characters(self):
        self.assertEqual(safe_name('cầu: rộng / "ban ngày"?'), "cầu rộng ban ngày")

    def test_empty_and_reserved(self):
        self.assertEqual(safe_name("   "), "")
        self.assertEqual(safe_name("_video_links"), "video_links")      # không đè file liên kết
        self.assertEqual(safe_name("(Mặc định của bạn)"), "Mặc định của bạn)")

    def test_source_key(self):
        self.assertEqual(source_key("D:/video/test1.mp4"), "test1.mp4")
        self.assertEqual(source_key("test1.mp4"), "test1.mp4")
        self.assertEqual(source_key("0"), "0")
        self.assertEqual(source_key("192.168.1.5:4747"), "192.168.1.5:4747")
        self.assertEqual(source_key("rtsp://cam/stream"), "rtsp://cam/stream")
        self.assertEqual(source_key("  "), "")


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PresetStore(Path(self.tmp.name) / "presets")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_load_roundtrip_with_meta(self):
        params = CalibParams(hfov_deg=72, pitch_deg=9.5, k=1.04)
        name = self.store.save("Cầu rộng", params, {"tuned_on": "a.mp4", "true_speed_kmh": 40})
        self.assertEqual(name, "Cầu rộng")
        loaded, meta = self.store.load("Cầu rộng")
        self.assertEqual((loaded.hfov_deg, loaded.pitch_deg, loaded.k), (72, 9.5, 1.04))
        self.assertEqual(meta["tuned_on"], "a.mp4")
        self.assertEqual(meta["true_speed_kmh"], 40)
        self.assertIn("saved_at", meta)

    def test_names_sorted_and_exclude_links_file(self):
        for n in ("beta", "Alpha", "gamma"):
            self.store.save(n, CalibParams())
        self.store.link_video("x.mp4", "beta")
        self.assertEqual(self.store.names(), ["Alpha", "beta", "gamma"])

    def test_overwrite_same_name(self):
        self.store.save("a", CalibParams(k=1.0))
        self.store.save("a", CalibParams(k=1.2))
        self.assertEqual(self.store.load("a")[0].k, 1.2)
        self.assertEqual(self.store.names(), ["a"])

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save("///", CalibParams())

    def test_missing_and_corrupt_return_none(self):
        self.assertIsNone(self.store.load("khong_co"))
        self.store.save("hong", CalibParams())
        (self.store.folder / "hong.json").write_text("{không phải json", encoding="utf-8")
        self.assertIsNone(self.store.load("hong"))

    def test_links(self):
        self.store.save("cau", CalibParams())
        self.store.link_video("a.mp4", "cau")
        self.store.link_video("b.mp4", "khong_ton_tai")                 # bộ không tồn tại: bỏ qua
        self.assertEqual(self.store.preset_for_video("a.mp4"), "cau")
        self.assertIsNone(self.store.preset_for_video("b.mp4"))
        self.assertIsNone(self.store.preset_for_video("c.mp4"))

    def test_delete_removes_links(self):
        self.store.save("cau", CalibParams())
        self.store.link_video("a.mp4", "cau")
        self.assertTrue(self.store.delete("cau"))
        self.assertFalse(self.store.delete("cau"))
        self.assertIsNone(self.store.preset_for_video("a.mp4"))
        self.assertEqual(self.store.names(), [])

    def test_empty_store(self):
        self.assertEqual(self.store.names(), [])
        self.assertIsNone(self.store.preset_for_video("a.mp4"))


class Warning(unittest.TestCase):
    def test_same_video_warns(self):
        meta = {"tuned_on": "test1.mp4"}
        self.assertIn("test1.mp4", same_video_warning(meta, "D:/video/test1.mp4"))

    def test_other_video_no_warning(self):
        self.assertIsNone(same_video_warning({"tuned_on": "test1.mp4"}, "test2.mp4"))
        self.assertIsNone(same_video_warning({}, "test2.mp4"))
        self.assertIsNone(same_video_warning(None, "test2.mp4"))


if __name__ == "__main__":
    unittest.main()