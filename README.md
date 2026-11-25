# 🚗 Ước lượng vận tốc xe từ camera đơn

**Phát hiện xe bằng YOLOv8, bám vết bằng ByteTrack, tính vận tốc (km/h) bằng hai phương pháp rồi so sánh với nhau.**

Đồ án môn Trí tuệ nhân tạo. Camera cố định quay từ phía sau xe, xe đi theo chiều dọc khung hình. Với mỗi xe, hệ thống đưa ra **hai** giá trị vận tốc:

| Phương pháp | Cách tính | Vai trò |
|---|---|---|
| **Mức 1: hai vạch** | Biết khoảng cách thật D giữa vạch A và B. v1 = D / (t₂ − t₁) × 3,6 | Chuẩn đối chứng, khá đáng tin |
| **Mức 2: kích thước xe** | Mét/pixel = bề rộng thật của loại xe / bề rộng khung xe. Vận tốc từ độ dời điểm bám | Học từ bài báo Tran et al. (2026), không cần đo đường |

Chạy song song để có thể **đo sai số** của từng mức so với tốc độ thật, và phân tích vì sao hai mức lệch nhau (hiệu ứng phối cảnh).

## Mục lục
[Cách hoạt động](#cách-hoạt-động) · [Cấu trúc thư mục](#cấu-trúc-thư-mục) · [Cài đặt](#cài-đặt) · [Chạy giao diện](#chạy-giao-diện) · [Chạy dòng lệnh và đánh giá sai số](#chạy-dòng-lệnh-và-đánh-giá-sai-số) · [Thiết lập vạch và khoảng cách](#thiết-lập-vạch-và-khoảng-cách) · [File CSV](#file-csv) · [Cấu hình](#cấu-hình) · [Kiểm thử](#kiểm-thử) · [Hạn chế](#giả-định-và-hạn-chế)

## Cách hoạt động

1. **Nhận diện và bám vết:** YOLOv8 (`weights/best.pt`, 4 lớp: car, motorbike, bus, truck) + ByteTrack gán ID cho từng xe.
2. **Điểm bám:** giữa cạnh dưới khung xe (chỗ xe chạm mặt đường), vì phép tính nằm trên mặt đường.
3. **Mức 1:** thời điểm xe cắt mỗi vạch được **nội suy tuyến tính giữa hai frame**, để không bị sai số làm tròn theo frame (khoảng ±1 frame ≈ 4% ở 40 km/h với video 30 FPS).
4. **Mức 2:** lấy **trung vị** của các tốc độ tức thời trong đoạn giữa hai vạch để bớt nhiễu. Bề rộng thật theo Table 1 của bài báo: xe máy 0,71 m, ô tô 1,74 m, bus 2,45 m, tải 2,0 m.
5. **Không xét hướng đi:** xe cắt vạch nào trước thì vạch đó là điểm xuất phát (cột `huong` ghi `A->B` hoặc `B->A`). Xe chỉ cắt một vạch (đổi làn, mất ID) bị bỏ qua.
6. **Lọc:** tốc độ Mức 1 ngoài khoảng 5 - 120 km/h được ghi vào CSV với `hop_le = 0` và không tính vào thống kê.
7. **Mốc thời gian:** file video dùng `số frame / FPS` (đúng cả khi giao diện bỏ bớt frame cho kịp); webcam / IP camera dùng đồng hồ thật, vì FPS của luồng mạng không ổn định.

> **Lưu ý về Mức 2.** Do phối cảnh, 1 pixel theo chiều dọc ứng với nhiều mét hơn 1 pixel theo chiều ngang khi xe ở xa. Mức 2 không hiệu chỉnh phối cảnh (bài báo xử lý bằng điểm triệt tiêu và hộp 3D, rất nặng) nên có thể **lệch hệ thống** so với Mức 1. Đây là kết quả thực nghiệm đáng ghi vào báo cáo.

## Cấu trúc thư mục

```text
speed_monitor/
├── main.py                       # Mở giao diện (demo, chỉnh vạch)
├── evaluate.py                   # Dòng lệnh: chạy trên video, xuất CSV + video có nhãn + sai số
├── requirements.txt              # Thư viện cần cài
├── README.md                     # Tài liệu này
├── .gitignore                    # Loại trừ video, log, cache khỏi Git
│
├── speed_monitor/                # Mã nguồn chính
│   ├── __init__.py               # Đánh dấu package, ghi phiên bản
│   ├── config.py                 # Toàn bộ tham số (vạch, khoảng cách, bảng kích thước xe, ngưỡng lọc, cổng IP)
│   ├── video_source.py           # Mở webcam / IP / file, đổi IP rút gọn thành URL, cấp mốc thời gian cho frame
│   ├── processor.py              # YOLO + ByteTrack, chỉ trả về danh sách xe (không biết gì về tốc độ)
│   ├── speed_estimator.py        # Tính tốc độ Mức 1 và Mức 2, lọc, thống kê, sai số (hàm thuần, dễ kiểm thử)
│   ├── pipeline.py               # Nối các bước cho một frame; giao diện và dòng lệnh cùng dùng
│   ├── drawing.py                # Vẽ vạch A/B, khung xe, nhãn km/h
│   ├── recorder.py               # Ghi CSV mỗi phiên
│   ├── camera_worker.py          # Luồng nền cho giao diện (không làm đơ cửa sổ)
│   └── app.py                    # Cửa sổ customtkinter duy nhất
│
├── tests/                        # Kiểm thử đơn vị (không cần model hay camera)
│   ├── test_speed.py             # Công thức, nội suy, lọc, thống kê, sai số
│   ├── test_source.py            # Đọc địa chỉ IP rút gọn, mốc thời gian video
│   └── test_pipeline.py          # Pipeline với bộ nhận diện giả, ghi CSV
│
├── weights/                      # Đặt best.pt (YOLOv8 đã huấn luyện) vào đây
├── video/                        # Video chạy thử và video có tốc độ thật (không đưa lên Git)
├── logs/                         # CSV và video kết quả của từng phiên
└── docs/                         # Ảnh chụp, kết quả huấn luyện, hình cho báo cáo
```

## Cài đặt

**Yêu cầu:** Python 3.10 trở lên; GPU NVIDIA khuyến nghị (không có thì chạy CPU, chậm hơn).

```powershell
pip install -r requirements.txt
# GPU NVIDIA: cài thêm PyTorch bản CUDA
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

Chép `best.pt` vào `weights/` và video của bạn vào `video/`.

## Chạy giao diện

```powershell
python main.py
```

1. Nhập nguồn video vào ô trên cùng, hoặc chọn từ danh sách "Video mẫu":

| Gõ | Nguồn |
|---|---|
| `0` | Webcam |
| `192.168.1.5` | IP camera: tự đổi thành `http://192.168.1.5:8080/video` |
| `192.168.1.5:4747` | Đổi cổng: `http://192.168.1.5:4747/video` |
| `http://...` hoặc `rtsp://...` | Giữ nguyên |
| `test1.mp4` | File trong `video/` |

   Cổng và đường dẫn mặc định (`8080`, `/video`, của ứng dụng IP Webcam trên Android) đổi được trong `config.py`.
2. Nhập **khoảng cách thật A - B (mét)** và kéo hai thanh trượt đặt vị trí vạch A, B.
3. Bấm **Start**. Trên xe hiện `L1` (hai vạch) và `L2` (kích thước xe); bảng bên dưới thống kê theo loại xe. Mỗi phiên ghi một file CSV vào `logs/`.

Chạy trực tiếp từ IP camera phù hợp để **demo**, vì độ trễ mạng làm lệch mốc thời gian. Để **đo sai số**, hãy quay video lưu thành file rồi chạy bằng `evaluate.py`.

## Chạy dòng lệnh và đánh giá sai số

```powershell
python evaluate.py --source test1.mp4 --distance 12 --line-a 30 --line-b 60 --true-speed 40
```

| Tham số | Ý nghĩa |
|---|---|
| `--source` | File trong `video/`, đường dẫn, IP camera hoặc `0` |
| `--distance` | Khoảng cách thật giữa hai vạch (mét) |
| `--line-a`, `--line-b` | Vị trí vạch, 0 = mép dưới, 100 = mép trên khung hình |
| `--true-speed` | Tốc độ thật (km/h) nếu biết, để tính sai số |
| `--max-frames` | Dừng sau N frame |
| `--no-video` | Không xuất video có nhãn |

Kết quả: file CSV và video có nhãn km/h trong `logs/`, cùng báo cáo trên màn hình (tốc độ trung bình, tỉ lệ Mức 2 / Mức 1, lệch trung bình và sai số tuyệt đối so với tốc độ thật).

**Cách có tốc độ thật để đánh giá:** quay video xe đi với tốc độ cố định đọc từ đồng hồ (ví dụ 30 hoặc 40 km/h) qua đoạn đường đã đo; quay vài video ở các tốc độ khác nhau rồi chạy `evaluate.py` cho từng video.

## Thiết lập vạch và khoảng cách

- Hai vạch là **hai đường ngang**, nên camera phải gần thẳng hàng với đường.
- Đo khoảng cách **D trên mặt đường** (vạch kẻ đường, cột mốc, hoặc Google Maps). D sai bao nhiêu phần trăm thì tốc độ Mức 1 sai bấy nhiêu phần trăm.
- Đặt hai vạch **đủ xa nhau** (nên cách nhau vài mét tới vài chục mét) và xa chỗ xe dừng chờ đèn đỏ.
- Đặt hai vạch ở vùng xe còn đủ lớn để nhận diện ổn định; xe ở rất xa thường bị mất ID.

## File CSV

Mỗi dòng là một xe đã đo xong (cắt đủ hai vạch):

| Cột | Ý nghĩa |
|---|---|
| `thoi_diem_s` | Thời điểm xe cắt vạch thứ hai (giây từ đầu phiên) |
| `id` | ID do ByteTrack gán |
| `loai_xe` | car / motorbike / bus / truck |
| `huong` | `A->B` hoặc `B->A` |
| `v_hai_vach_kmh` | Tốc độ Mức 1 |
| `v_kich_thuoc_kmh` | Tốc độ Mức 2 (để trống nếu không tính được) |
| `hop_le` | 1 nếu Mức 1 nằm trong 5 - 120 km/h, ngược lại 0 |

Mở bằng Excel để vẽ biểu đồ so sánh hai mức cho báo cáo.

## Cấu hình

Mọi tham số nằm trong `speed_monitor/config.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `DEFAULT_LINE_A_PCT`, `DEFAULT_LINE_B_PCT` | 30, 60 | Vị trí hai vạch |
| `DEFAULT_DISTANCE_M` | 10 | Khoảng cách thật giữa hai vạch (m) |
| `VEHICLE_WIDTH_M` | 0,71 / 1,74 / 2,45 / 2,0 | Bề rộng thật: xe máy / ô tô / bus / tải |
| `WIDTH_WINDOW` | 5 | Số mẫu cách nhau khi tính tốc độ tức thời Mức 2 |
| `MIN_VALID_KMH`, `MAX_VALID_KMH` | 5, 120 | Khoảng tốc độ hợp lệ |
| `MAX_GAP_S` | 1,0 | Mất dấu xe lâu hơn mức này thì không nội suy qua khoảng đó |
| `CONFIDENCE_THRESHOLD` | 0,35 | Ngưỡng tự tin của YOLO |
| `PROCESS_WIDTH` | 960 | Thu nhỏ khung hình về chiều rộng này trước khi xử lý |
| `IP_DEFAULT_PORT`, `IP_DEFAULT_PATH` | 8080, `/video` | Mặc định khi gõ IP rút gọn |
| `LOOP_VIDEO_FILES` | False | True = video file tự phát lại từ đầu |

## Kiểm thử

```powershell
python -m unittest discover -s tests -t . -v
```

Các test không cần model hay camera: công thức tốc độ với số giả (biết trước đáp án), nội suy, lọc giá trị phi lý, mốc thời gian video, cách đọc IP rút gọn, pipeline với bộ nhận diện giả và ghi CSV.

## Giả định và hạn chế

- Camera cố định, vạch là đường ngang, xe đi gần song song với trục dọc khung hình.
- Mức 1 chính xác phụ thuộc vào việc đo D và vào độ ổn định của khung xe (khung dao động làm lệch điểm bám).
- Mức 2 giả định mọi xe cùng loại có cùng bề rộng và bỏ qua phối cảnh theo chiều sâu, nên có thể lệch hệ thống so với Mức 1.
- Xe bị che khuất hoặc đổi ID giữa hai vạch sẽ không được đo.
- Chạy qua mạng (IP camera) có độ trễ không ổn định, không phù hợp để đánh giá sai số.

## Tài liệu tham khảo

Tran et al., *Traffic flow velocity estimation by virtually calibrated monocular surveillance cameras*, Discover Artificial Intelligence, 2026, 6:273. Nguồn của bảng kích thước xe (Table 1) và ý tưởng ước lượng vận tốc từ camera đơn.
