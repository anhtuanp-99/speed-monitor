"""Dòng lệnh: chạy trên video, xuất CSV + video có nhãn + báo cáo tốc độ (và sai số nếu biết tốc độ thật).

Hệ số hiệu chỉnh lấy từ calibration.json (cùng file giao diện lưu), có thể ghi đè bằng tham số dòng lệnh.

Ví dụ:
    python evaluate.py --source test1.mp4 --true-speed 40
    python evaluate.py --source test1.mp4 --true-speed 40 --pitch 9 --hfov 70 --k 1.05
"""
import argparse
import statistics

import cv2

from speed_monitor import config
from speed_monitor.camera_model import CalibParams
from speed_monitor.pipeline import SpeedPipeline
from speed_monitor.recorder import Recorder
from speed_monitor.speed_estimator import error_stats
from speed_monitor.video_source import VideoStream, resize_to_width, resolve_source


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ước lượng tốc độ xe từ camera đơn trên video")
    p.add_argument("--source", required=True, help="File trong video/, đường dẫn, IP camera hoặc 0")
    p.add_argument("--true-speed", type=float, default=None, help="Tốc độ thật (km/h) để tính sai số")
    p.add_argument("--max-frames", type=int, default=None, help="Dừng sau N frame")
    p.add_argument("--no-video", action="store_true", help="Không xuất video có nhãn")
    p.add_argument("--calib", default=None, help="File hệ số (mặc định calibration.json)")
    p.add_argument("--hfov", type=float, default=None, help="Góc nhìn ngang (độ)")
    p.add_argument("--pitch", type=float, default=None, help="Góc cúi camera (độ)")
    p.add_argument("--height", type=float, default=None, help="Độ cao camera (m), dùng cùng --manual-height")
    p.add_argument("--manual-height", action="store_true", help="Tắt tự động độ cao H, dùng --height / giá trị lưu")
    p.add_argument("--k", type=float, default=None, help="Hệ số hiệu chỉnh chung")
    p.add_argument("--zone-top", type=float, default=None, help="Cạnh trên vùng đo (%% từ mép dưới)")
    p.add_argument("--zone-bottom", type=float, default=None, help="Cạnh dưới vùng đo (%% từ mép dưới)")
    return p.parse_args(argv)


def build_params(args):
    params = CalibParams.load(args.calib)
    for attr, value in (("hfov_deg", args.hfov), ("pitch_deg", args.pitch), ("height_m", args.height),
                        ("k", args.k), ("zone_top_pct", args.zone_top),
                        ("zone_bottom_pct", args.zone_bottom)):
        if value is not None:
            setattr(params, attr, value)
    if args.manual_height:
        params.auto_height = False
    return params.clamped()


def main(argv=None):
    args = parse_args(argv)
    params = build_params(args)
    recorder = Recorder()
    pipeline = SpeedPipeline(recorder=recorder, params=params)
    pipeline.reset()
    stream = VideoStream(resolve_source(args.source))
    writer, video_path, frames = None, recorder.path.with_suffix(".mp4"), 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame, t = stream.read()
            if not ok:
                break
            frame = resize_to_width(frame, config.PROCESS_WIDTH)
            annotated, _ = pipeline.process(frame, t)
            if not args.no_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             stream.fps or config.DEFAULT_FPS, (w, h))
                writer.write(annotated)
            frames += 1
    finally:
        stream.release()
        if writer is not None:
            writer.release()
        recorder.close()

    info = pipeline.info()
    valid = [r for r in pipeline.records if r.valid]
    print(f"\nĐã xử lý {frames} frame. CSV: {recorder.path}")
    if writer is not None:
        print(f"Video có nhãn: {video_path}")
    mode = "tự động" if info["auto"] else "nhập tay"
    print(f"Độ cao camera H = {info['height_m']:.2f} m ({mode}, {info['votes']} xe để tự tính); "
          f"hfov {params.hfov_deg:g}°, góc cúi {params.pitch_deg:g}°, k = {params.k:g}")
    print(f"Đo được {len(valid)} xe hợp lệ, {len(pipeline.records) - len(valid)} bị loại "
          f"(> {config.MAX_VALID_KMH:g} km/h).")
    for name, speeds in sorted(pipeline.stats.by_class.items()):
        print(f"  {name}: {len(speeds)} xe, TB {statistics.mean(speeds):.1f} km/h "
              f"(min {min(speeds):.0f}, max {max(speeds):.0f})")
    flow = pipeline.stats.flow()
    if flow:
        print(f"Dòng xe: TB {flow['mean']:.1f} km/h, tình trạng: {flow['state']}")
    for name, reasons in sorted(pipeline.stats.lost.items()):
        print(f"  {name} không đo được: {reasons}")
    if args.true_speed:
        stats = error_stats(pipeline.records, args.true_speed)["v"]
        if stats is None:
            print("Chưa có xe hợp lệ để tính sai số.")
        else:
            print(f"So với tốc độ thật {args.true_speed:g} km/h ({stats['n']} xe): "
                  f"lệch TB {stats['bias']:+.1f} km/h, sai số tuyệt đối TB {stats['mean_abs']:.1f} km/h "
                  f"({stats['mean_pct']:.1f}%), trung vị {stats['median_abs']:.1f} km/h")


if __name__ == "__main__":
    main()