"""Kiểm thử cách đọc địa chỉ nguồn video (không cần camera)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from speed_monitor import config, video_source
from speed_monitor.video_source import is_file_source, resolve_source


class TestResolveSource(unittest.TestCase):
    def test_webcam(self):
        self.assertEqual(resolve_source("0"), 0)
        self.assertEqual(resolve_source(" 1 "), 1)

    def test_ip_shorthand_uses_defaults(self):
        self.assertEqual(resolve_source("192.168.1.5"),
                         f"http://192.168.1.5:{config.IP_DEFAULT_PORT}{config.IP_DEFAULT_PATH}")

    def test_ip_with_custom_port(self):
        self.assertEqual(resolve_source("192.168.1.5:4747"),
                         f"http://192.168.1.5:4747{config.IP_DEFAULT_PATH}")

    def test_ip_with_custom_port_and_path(self):
        self.assertEqual(resolve_source("10.0.0.2:8080/videofeed"),
                         "http://10.0.0.2:8080/videofeed")

    def test_full_url_kept(self):
        self.assertEqual(resolve_source("http://192.168.1.5:8080/video"),
                         "http://192.168.1.5:8080/video")
        self.assertEqual(resolve_source("rtsp://cam/stream"), "rtsp://cam/stream")

    def test_invalid_ip_is_treated_as_file(self):
        self.assertFalse(resolve_source("999.1.1.1").startswith("http://"))

    def test_file_found_in_video_dir(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "clip.mp4").write_bytes(b"x")
            with mock.patch.object(config, "VIDEO_DIR", Path(tmp)):
                self.assertEqual(resolve_source("clip.mp4"), str(Path(tmp) / "clip.mp4"))

    def test_is_file_source(self):
        self.assertTrue(is_file_source("clip.mp4"))
        self.assertFalse(is_file_source("http://1.2.3.4:8080/video"))
        self.assertFalse(is_file_source(0))


class TestVideoStreamClock(unittest.TestCase):
    def test_file_timestamps_follow_frame_count(self):
        class FakeCap:
            def __init__(self):
                self.n = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 25.0

            def set(self, *a):
                return True

            def read(self):
                self.n += 1
                return True, object()

            def grab(self):
                return True

            def release(self):
                pass

        with mock.patch.object(video_source, "open_video_capture", return_value=FakeCap()):
            s = video_source.VideoStream("clip.mp4")
            _, _, t0 = s.read()
            s.skip(3)                       # frame bỏ qua vẫn được tính vào thời gian
            _, _, t1 = s.read()
        self.assertAlmostEqual(t0, 0.0)
        self.assertAlmostEqual(t1, 4 / 25.0)


if __name__ == "__main__":
    unittest.main()
