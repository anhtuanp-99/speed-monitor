"""Cấu hình tập trung. Các hệ số hiệu chỉnh camera chỉnh trực tiếp trên giao diện (tab "Hệ số")
và được lưu vào calibration.json; file này chỉ chứa giá trị mặc định và các hằng số kỹ thuật."""
from datetime import datetime
from pathlib import Path

try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


def describe_device():
    """Mô tả thiết bị đang dùng để hiển thị cho người dùng (GPU nào, hay đang chạy CPU)."""
    if DEVICE.startswith("cuda"):
        import torch
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU (chậm: PyTorch chưa nhận GPU, xem mục Kiểm tra GPU trong README)"

# ==============================================================================
# 1. ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent      # thư mục gốc dự án
MODEL_PATH = BASE_DIR / "weights" / "best.pt"
VIDEO_DIR = BASE_DIR / "video"
LOG_DIR = BASE_DIR / "logs"
CALIB_FILE = BASE_DIR / "calibration.json"             # hệ số hiệu chỉnh đã lưu


def day_log_dir(now=None):
    """Thư mục log của ngày hiện tại: logs/YYYY-MM-DD/ (tự tạo nếu chưa có)."""
    folder = LOG_DIR / (now or datetime.now()).strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# ==============================================================================
# 2. MÔ HÌNH & TRACKING
# ==============================================================================
# Chỉ dùng làm dự phòng: tên lớp thật được lấy từ model.names của best.pt (xem processor.py).
CLASS_NAMES = {0: "car", 1: "motorbike", 2: "bus", 3: "truck"}

# Màu BGR của khung nhận diện theo loại xe
CLASS_COLORS = {
    "car": (0, 255, 0),
    "motorbike": (0, 165, 255),
    "bus": (255, 0, 255),
    "truck": (255, 128, 0),
}

TRACKER_TYPE = "bytetrack.yaml"
# Ngưỡng thấp để ByteTrack còn nhận được các khung điểm thấp (xe ở xa, bị che) và nối ID tốt hơn.
DETECT_CONF = 0.15
CONFIDENCE_THRESHOLD = DETECT_CONF   # tên cũ, giữ để mã cũ không lỗi
AGNOSTIC_NMS = True                  # một xe chỉ giữ một khung (tránh car + truck chồng nhau)
PROCESS_WIDTH = 960              # khung hình được thu nhỏ về chiều rộng này trước khi xử lý

# ==============================================================================
# 3. HỆ SỐ HIỆU CHỈNH MẶC ĐỊNH (chỉnh trên giao diện, xem HUONG_DAN_HE_SO.md)
# ==============================================================================
# Bề rộng thật của xe (mét) theo Table 1 của Tran et al. (2026): "thước đo" tỉ lệ của mô hình.
VEHICLE_WIDTH_M = {
    "motorbike": 0.71,
    "car": 1.74,
    "bus": 2.45,
    "truck": 2.0,
}
DEFAULT_HFOV_DEG = 65.0          # góc nhìn ngang (điện thoại, camera chính, 16:9 thường 60 - 70)
DEFAULT_PITCH_DEG = 10.0         # góc cúi của camera so với phương ngang (căn bằng đường chân trời)
DEFAULT_HEIGHT_M = 7.0           # độ cao camera, dùng khi tắt chế độ tự động hoặc chưa đủ xe để tự tính
DEFAULT_K = 1.0                  # hệ số hiệu chỉnh chung nhân vào tốc độ (hệ số thực nghiệm)
# Vùng đo theo % chiều cao khung hình: 0 = mép dưới, 100 = mép trên. Chỉ đo xe có điểm bám trong vùng
# (xa quá thì xe quá nhỏ, sai số lớn).
DEFAULT_ZONE_TOP_PCT = 40.0      # cạnh trên (xa camera) của vùng đo
DEFAULT_ZONE_BOTTOM_PCT = 8.0    # cạnh dưới (gần camera)

# Giới hạn hợp lệ của các hệ số (giao diện và file lưu đều bị kẹp trong khoảng này)
LIMITS = {
    "hfov_deg": (20.0, 140.0),
    "pitch_deg": (0.0, 80.0),
    "height_m": (0.5, 100.0),
    "k": (0.2, 3.0),
    "zone_top_pct": (0.0, 100.0),
    "zone_bottom_pct": (0.0, 100.0),
    "width_m": (0.3, 5.0),
}

# ==============================================================================
# 4. ƯỚC LƯỢNG TỐC ĐỘ
# ==============================================================================
MIN_DENOM = 0.05                 # điểm ảnh quá sát đường chân trời (mẫu số hình học nhỏ hơn mức này) bị bỏ qua
MIN_SAMPLES = 8                  # tối thiểu số mẫu trong vùng đo để ra kết quả cuối cùng của một xe
MIN_SPAN_S = 0.5                 # và tối thiểu thời gian (giây) xe ở trong vùng đo
MAX_GAP_S = 1.0                  # mất dấu xe lâu hơn mức này thì bỏ các mẫu cũ, đo lại từ đầu
MAX_SAMPLES_PER_TRACK = 900      # giới hạn số mẫu lưu cho mỗi xe (xe đứng lâu trong vùng đo không làm tràn bộ nhớ)
STALE_AFTER_S = 3.0              # xe vắng mặt quá lâu thì chốt kết quả (nếu đủ mẫu) và xóa trạng thái
STOP_KMH = 2.0                   # dưới mức này coi như xe đứng, ghi 0 (khung nhận diện luôn rung nhẹ)
MAX_VALID_KMH = 120.0            # nhanh hơn mức này coi là không hợp lệ (khung nhảy, mất ID)

# Tự tính độ cao camera H từ bề rộng khung xe (trung vị trên nhiều xe)
AUTO_HEIGHT_MIN_TRACKS = 8       # cần ít nhất ngần này xe đã đo xong mới dùng H tự động
AUTO_HEIGHT_MAX_TRACKS = 60      # chỉ giữ các xe gần nhất

# Tốc độ hiện trực tiếp trên khung xe khi đang ở trong vùng đo
LIVE_WINDOW_S = 1.0              # tính trên các mẫu trong 1 giây gần nhất
LIVE_MIN_SAMPLES = 6
LIVE_MIN_SPAN_S = 0.3
LIVE_SMOOTHING = 0.4             # trọng số của giá trị mới khi làm mượt (0 - 1; nhỏ = mượt hơn)
MAX_FIT_POINTS = 60              # nhiều mẫu hơn mức này thì lấy thưa ra cho nhanh

# Tình trạng dòng xe theo tốc độ trung bình (km/h): dưới ngưỡng 1 = kẹt, dưới ngưỡng 2 = đông
TRAFFIC_THRESHOLDS = (15.0, 30.0)

# ==============================================================================
# 5. HIỂN THỊ NHÃN TRÊN KHUNG XE
# ==============================================================================
SPEED_LABEL_COLOR = (0, 0, 255)        # BGR: đỏ, nhãn tốc độ
SPEED_LABEL_OUTLINE = (255, 255, 255)  # viền trắng cho dễ đọc trên mọi nền
MEASURING_TEXT = "dang do..."          # xe đang trong vùng đo nhưng chưa đủ mẫu

# ==============================================================================
# 6. NGUỒN VIDEO
# ==============================================================================
# Gõ "192.168.1.5" hoặc "192.168.1.5:4747" sẽ tự đổi thành http://IP:PORT/PATH.
# Mặc định của ứng dụng IP Webcam (Android): cổng 8080, đường dẫn /video.
IP_DEFAULT_PORT = 8080
IP_DEFAULT_PATH = "/video"
DEFAULT_FPS = 30.0               # dùng khi nguồn không báo FPS hợp lệ

# ==============================================================================
# 7. GIAO DIỆN
# ==============================================================================
LOOP_VIDEO_FILES = False         # video file chạy hết thì dừng (True = phát lại từ đầu)
# 0 = không bỏ frame nào (tracker thấy đủ frame nên không đổi ID). Máy yếu sẽ chạy chậm hơn thực tế
# nhưng tốc độ đo vẫn đúng vì mốc thời gian tính theo số frame / FPS.
MAX_FRAME_SKIP = 0
VIDEO_PLACEHOLDER_W = 640
VIDEO_PLACEHOLDER_H = 360
APPEARANCE_MODE = "light"        # "light", "dark" hoặc "system"
UI_REFRESH_MS = 30
WINDOW_TITLE = "Ước lượng vận tốc xe từ camera đơn (YOLOv8 + ByteTrack + mô hình mặt đường)"
RECENT_SHOWN = 12                # số kết quả đo gần nhất hiển thị trên giao diện