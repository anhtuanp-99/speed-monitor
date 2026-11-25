"""Các hàm vẽ lên khung hình (cv2.putText không hỗ trợ dấu tiếng Việt nên dùng không dấu)."""
import cv2

from . import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_COLOR = (0, 255, 255)
HORIZON_COLOR = (255, 255, 0)


def _outlined_text(frame, text, org, scale, color, thickness=2):
    cv2.putText(frame, text, org, FONT, scale, config.SPEED_LABEL_OUTLINE, thickness + 3)
    cv2.putText(frame, text, org, FONT, scale, color, thickness)


def draw_zone(frame, zone_rows, horizon_row, pitch_deg, ground_range):
    """Vẽ vùng đo (băng mờ viền vàng) và đường chân trời (nét đứt xanh lơ) để căn góc cúi camera."""
    h, w = frame.shape[:2]
    if horizon_row is not None and 0 <= horizon_row < h:
        y = int(horizon_row)
        for x in range(0, w, 24):
            cv2.line(frame, (x, y), (min(x + 12, w), y), HORIZON_COLOR, 2)
        cv2.putText(frame, f"CHAN TROI (goc cui {pitch_deg:.1f} do)", (10, max(y - 6, 14)),
                    FONT, 0.5, HORIZON_COLOR, 1)
    if zone_rows is None:
        return
    top, bottom = int(zone_rows[0]), int(zone_rows[1])
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (w, bottom), ZONE_COLOR, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.line(frame, (0, top), (w, top), ZONE_COLOR, 2)
    cv2.line(frame, (0, bottom), (w, bottom), ZONE_COLOR, 2)
    label = "VUNG DO"
    if ground_range is not None:
        label += f" (cach cam {ground_range[0]:.0f} - {ground_range[1]:.0f} m)"
    cv2.putText(frame, label, (10, min(bottom - 8, h - 8)), FONT, 0.55, ZONE_COLOR, 2)


def draw_vehicle(frame, det, record=None, live_kmh=None, measuring=False):
    """Vẽ khung xe, ID (đếm từ 1) và nhãn tốc độ màu đỏ:
    - đã chốt kết quả (xe đã rời vùng đo): tốc độ cuối cùng, chữ lớn
    - đang trong vùng đo: tốc độ tức thời; chưa đủ mẫu thì hiện "dang do..." """
    color = config.CLASS_COLORS.get(det.class_name, (0, 255, 0))
    x1, y1 = int(det.cx - det.w / 2), int(det.cy - det.h / 2)
    x2, y2 = int(det.cx + det.w / 2), int(det.cy + det.h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (int(det.anchor_x), int(det.anchor_y)), 3, color, -1)
    cv2.putText(frame, f"ID:{det.track_id} {det.class_name}", (x1, max(y1 - 30, 12)),
                FONT, 0.45, color, 1)

    org = (x1, max(y1 - 8, 12))
    if record is not None:
        text = f"{record.speed_kmh:.0f} km/h"
        if not record.valid:
            text += " (loai)"
        _outlined_text(frame, text, org, 0.7, config.SPEED_LABEL_COLOR)
    elif live_kmh is not None:
        _outlined_text(frame, f"{live_kmh:.0f} km/h", org, 0.55, config.SPEED_LABEL_COLOR, 1)
    elif measuring:
        _outlined_text(frame, config.MEASURING_TEXT, org, 0.5, (0, 140, 255), 1)


def draw_info(frame, measured, height_m, auto, votes):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (330, 76), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Xe da do: {measured}", (16, 28), FONT, 0.5, (255, 255, 255), 1)
    mode = f"tu dong, {votes} xe" if auto else f"nhap tay ({votes} xe de tu tinh)"
    cv2.putText(frame, f"Do cao camera H = {height_m:.1f} m ({mode})", (16, 48),
                FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "Nhan do = toc do (km/h)", (16, 68), FONT, 0.45, (200, 200, 200), 1)