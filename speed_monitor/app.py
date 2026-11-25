"""Cửa sổ duy nhất: cột trái = nguồn, tab Log, tab Hệ số (chỉnh tay); cột phải = video (chiếm phần lớn)."""
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk

from . import config
from .camera_model import CalibParams
from .camera_worker import CameraWorker

STATE_TEXT = {"idle": "Chưa chạy", "loading": "Đang nạp mô hình...", "stopped": "Đã dừng"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_PLACEHOLDER = "Video mẫu"
CLASS_LABELS = {"motorbike": "Xe máy", "car": "Ô tô", "bus": "Bus", "truck": "Tải"}
DIRECTION_LABELS = {"ra_xa": "đi xa", "lai_gan": "lại gần", "dung": "đứng"}
LOST_LABELS = {"it_mau": "quá ít mẫu trong vùng đo"}
STATE_LABELS = {"thong thoang": "Thông thoáng", "dong": "Đông", "ket": "Kẹt"}
LEFT_PANEL_W = 400
TAB_LOG, TAB_PARAMS = "Log", "Hệ số"

# (khóa, nhãn, nhỏ nhất, lớn nhất, số bước, định dạng, gợi ý) theo thứ tự ảnh hưởng tới tốc độ
SLIDERS = [
    ("k", "Hệ số chung k", 0.5, 1.5, 100, "{:.2f}", "tốc độ ∝ k (tăng 10% -> tốc độ tăng 10%)"),
    ("height_m", "Độ cao camera H (khi nhập tay)", 1.0, 40.0, 390, "{:.1f} m", "tốc độ ∝ H"),
    ("pitch_deg", "Góc cúi camera", 0.0, 60.0, 240, "{:.1f}°", "căn đường chân trời xanh lơ cho khớp ảnh"),
    ("hfov_deg", "Góc nhìn ngang HFOV", 30.0, 120.0, 90, "{:.0f}°", "điện thoại thường 60 - 70°"),
    ("zone_top_pct", "Vùng đo - cạnh trên (xa)", 0.0, 100.0, 100, "{:.0f}%", "0 = mép dưới khung hình"),
    ("zone_bottom_pct", "Vùng đo - cạnh dưới (gần)", 0.0, 100.0, 100, "{:.0f}%", "xe ở xa nhỏ nên sai số lớn"),
]


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode(config.APPEARANCE_MODE)
        super().__init__()
        self.title(config.WINDOW_TITLE)
        self.geometry("1450x820")
        self.minsize(1050, 640)
        self.params = CalibParams.load()
        self.worker = CameraWorker(self.params.copy())
        self._last_frame_id = -1
        self._photo = None
        self._last_log_text = None
        self._sliders = {}
        self._width_entries = {}

        # Cột 0 = panel trái (cố định), cột 1 = video (giãn hết phần còn lại)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_video()
        self._sync_widgets()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._after_id = self.after(config.UI_REFRESH_MS, self._tick)

    # ---------------- dựng giao diện ----------------
    def _build_left_panel(self):
        left = ctk.CTkFrame(self, width=LEFT_PANEL_W)
        left.grid(row=0, column=0, sticky="ns", padx=(10, 4), pady=10)
        left.grid_propagate(False)                    # giữ đúng bề rộng, không co giãn theo nội dung
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)           # tab chiếm phần cao còn lại

        self._build_source_row(left)
        self.status_label = ctk.CTkLabel(left, text="Chưa chạy", anchor="w", justify="left",
                                         wraplength=LEFT_PANEL_W - 24)
        self.status_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

        self.tabs = ctk.CTkTabview(left)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 8))
        self.tabs.add(TAB_LOG)
        self.tabs.add(TAB_PARAMS)
        self.log_box = ctk.CTkTextbox(self.tabs.tab(TAB_LOG), wrap="none", font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")
        self._build_params_tab(self.tabs.tab(TAB_PARAMS))

    def _build_source_row(self, parent):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=0, sticky="ew", padx=6, pady=(10, 4))
        box.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            box, placeholder_text="0 (webcam) | 192.168.x.x | tên file video")
        self.entry.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 6))

        videos = sorted(p.name for p in config.VIDEO_DIR.glob("*")
                        if p.suffix.lower() in VIDEO_SUFFIXES) if config.VIDEO_DIR.exists() else []
        self.sample_menu = ctk.CTkOptionMenu(
            box, values=videos or ["(chưa có video)"], command=self._on_pick_sample)
        self.sample_menu.set(SAMPLE_PLACEHOLDER)
        self.sample_menu.grid(row=1, column=0, sticky="ew", padx=4)

        self.button = ctk.CTkButton(box, text="Start", width=90, command=self._toggle)
        self.button.grid(row=1, column=1, padx=4)

    def _build_video(self):
        box = tk.Frame(self, bg="black")
        box.grid(row=0, column=1, sticky="nsew", padx=(4, 10), pady=10)
        self.video_label = tk.Label(box, bg="black", fg="#9a9a9a", text="Chưa có hình", bd=0)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        box.bind("<Configure>", lambda e: self.worker.set_display_size(e.width, e.height))

    def _build_params_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll.grid_columnconfigure(0, weight=1)
        row = 0

        self.info_label = ctk.CTkLabel(scroll, text="", anchor="w", justify="left",
                                       wraplength=LEFT_PANEL_W - 60, font=ctk.CTkFont(weight="bold"))
        self.info_label.grid(row=row, column=0, sticky="ew", padx=4, pady=(0, 6))
        row += 1

        self.auto_switch = ctk.CTkSwitch(scroll, text="Tự động độ cao H (từ bề rộng xe)",
                                         command=self._on_auto)
        self.auto_switch.grid(row=row, column=0, sticky="w", padx=4, pady=(0, 6))
        row += 1

        for key, label, low, high, steps, fmt, hint in SLIDERS:
            title = ctk.CTkLabel(scroll, text="", anchor="w")
            title.grid(row=row, column=0, sticky="ew", padx=4)
            slider = ctk.CTkSlider(scroll, from_=low, to=high, number_of_steps=steps,
                                   command=lambda v, k=key: self._on_slider(k, v))
            slider.grid(row=row + 1, column=0, sticky="ew", padx=4)
            ctk.CTkLabel(scroll, text=hint, anchor="w", text_color=("#666666", "#aaaaaa"),
                         font=ctk.CTkFont(size=11)).grid(row=row + 2, column=0, sticky="ew", padx=4,
                                                         pady=(0, 6))
            self._sliders[key] = (slider, title, label, fmt)
            row += 3

        ctk.CTkLabel(scroll, text="Bề rộng thật của xe (m) - thước đo tỉ lệ", anchor="w",
                     font=ctk.CTkFont(weight="bold")).grid(row=row, column=0, sticky="ew", padx=4,
                                                           pady=(4, 2))
        row += 1
        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.grid(row=row, column=0, sticky="ew", padx=4)
        for i, name in enumerate(config.VEHICLE_WIDTH_M):
            ctk.CTkLabel(grid, text=CLASS_LABELS.get(name, name), width=60, anchor="w").grid(
                row=i // 2, column=(i % 2) * 2, padx=(0, 4), pady=2)
            entry = ctk.CTkEntry(grid, width=60)
            entry.grid(row=i // 2, column=(i % 2) * 2 + 1, padx=(0, 14), pady=2)
            entry.bind("<Return>", lambda e: self._apply_widths())
            entry.bind("<FocusOut>", lambda e: self._apply_widths())
            self._width_entries[name] = entry
        row += 1
        self.note_label = ctk.CTkLabel(scroll, text="", anchor="w", text_color=("#b00020", "#ff8a80"))
        self.note_label.grid(row=row, column=0, sticky="ew", padx=4)
        row += 1

        buttons = ctk.CTkFrame(scroll, fg_color="transparent")
        buttons.grid(row=row, column=0, sticky="ew", padx=4, pady=(6, 10))
        ctk.CTkButton(buttons, text="Lưu hệ số", width=110, command=self._save).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(buttons, text="Mặc định", width=110, fg_color="gray50",
                      command=self._reset_defaults).grid(row=0, column=1)

    # ---------------- đồng bộ giữa hệ số và giao diện ----------------
    def _sync_widgets(self):
        p = self.params
        for key, (slider, title, label, fmt) in self._sliders.items():
            slider.set(getattr(p, key))
            title.configure(text=f"{label}: {fmt.format(getattr(p, key))}")
        if p.auto_height:
            self.auto_switch.select()
        else:
            self.auto_switch.deselect()
        for name, entry in self._width_entries.items():
            entry.delete(0, "end")
            entry.insert(0, f"{p.widths.get(name, config.VEHICLE_WIDTH_M[name]):g}")
        self.worker.set_params(p)

    def _push(self):
        self.params = self.params.clamped()
        self.worker.set_params(self.params)

    def _on_slider(self, key, value):
        setattr(self.params, key, float(value))
        _slider, title, label, fmt = self._sliders[key]
        title.configure(text=f"{label}: {fmt.format(float(value))}")
        self._push()

    def _on_auto(self):
        self.params.auto_height = bool(self.auto_switch.get())
        self._push()

    def _apply_widths(self):
        new = {}
        for name, entry in self._width_entries.items():
            try:
                value = float(entry.get().replace(",", "."))
                low, high = config.LIMITS["width_m"]
                if not low <= value <= high:
                    raise ValueError
                new[name] = value
            except ValueError:
                self.note_label.configure(text=f"Bề rộng phải nằm trong {config.LIMITS['width_m'][0]:g} - "
                                               f"{config.LIMITS['width_m'][1]:g} m")
                return
        self.note_label.configure(text="")
        self.params.widths.update(new)
        self._push()

    def _save(self):
        self._apply_widths()
        path = self.params.save()
        self.note_label.configure(text=f"Đã lưu: {path.name}", text_color=("#1b5e20", "#a5d6a7"))

    def _reset_defaults(self):
        self.params = CalibParams()
        self._sync_widgets()
        self.note_label.configure(text="Đã về mặc định (chưa lưu)", text_color=("#666666", "#aaaaaa"))

    # ---------------- sự kiện ----------------
    def _on_pick_sample(self, name):
        if name.startswith("("):
            return
        self.entry.delete(0, "end")
        self.entry.insert(0, name)
        self.sample_menu.set(SAMPLE_PLACEHOLDER)

    def _toggle(self):
        if self.worker.is_active:
            self.worker.stop()
            return
        source = self.entry.get().strip()
        if not source:
            self.status_label.configure(text="Hãy nhập nguồn video (file, IP camera hoặc 0).")
            return
        self._apply_widths()
        self.worker.start(source)

    def _on_close(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.worker.stop()
        self.worker.join(3.0)
        self.destroy()

    # ---------------- cập nhật định kỳ ----------------
    def _tick(self):
        self._refresh()
        self._after_id = self.after(config.UI_REFRESH_MS, self._tick)

    def _set_log(self, text):
        """Chỉ cập nhật khi nội dung đổi, và giữ nguyên vị trí cuộn của người dùng."""
        if text == self._last_log_text:
            return
        self._last_log_text = text
        y = self.log_box.yview()[0]
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", text)
        self.log_box.configure(state="disabled")
        self.log_box.yview_moveto(y)

    def _info_text(self, info):
        if not info:
            return ""
        mode = f"tự động từ {info['votes']} xe" if info["auto"] else \
            f"nhập tay (mới có {info['votes']} xe để tự tính)"
        text = f"H đang dùng: {info['height_m']:.1f} m ({mode})"
        if info.get("zone_range"):
            near, far = info["zone_range"]
            text += f"\nVùng đo cách camera {near:.0f} - {far:.0f} m (dài ≈ {far - near:.0f} m)"
        return text

    def _refresh(self):
        st = self.worker.status
        if st.state == "running":
            text = f"Đang chạy · {st.fps:.0f} FPS · xe trong khung: {st.active}"
        elif st.state == "error":
            text = f"Lỗi: {st.message}"
        else:
            text = STATE_TEXT.get(st.state, st.state)
        if self.worker.is_stopping:
            text = "Đang dừng..."
        if self.worker.device_text:
            text += f"\n{self.worker.device_text}"
        if st.csv_path:
            text += f"\nCSV: {st.csv_path}"
        self.status_label.configure(text=text)
        self.button.configure(
            text="Stop" if self.worker.is_active and not self.worker.is_stopping else "Start")
        self.info_label.configure(text=self._info_text(st.info))

        if self.worker.frame_id != self._last_frame_id and self.worker.latest_frame is not None:
            self._last_frame_id = self.worker.frame_id
            self._photo = ImageTk.PhotoImage(Image.fromarray(self.worker.latest_frame))
            self.video_label.configure(image=self._photo, text="")

        lines = [f"Đã đo: {st.measured} xe hợp lệ ({st.invalid} bị loại)"]
        if st.flow:
            lines.append(f"Dòng xe: TB {st.flow['mean']:.1f} km/h - "
                         f"{STATE_LABELS.get(st.flow['state'], st.flow['state'])}")
        lines.append("")
        for name, s in st.summary.items():
            lines.append(f"{CLASS_LABELS.get(name, name)}: {s['n']} xe")
            lines.append(f"   TB {s['mean']:.1f} | min {s['min']:.0f} | max {s['max']:.0f} km/h")
        if st.lost:
            lines.append("")
            lines.append("Không đo được:")
            for name, reasons in st.lost.items():
                detail = ", ".join(f"{LOST_LABELS.get(r, r)}: {n}" for r, n in reasons.items())
                lines.append(f"   {CLASS_LABELS.get(name, name)} - {detail}")
        if st.recent:
            lines.append("")
            lines.append("Gần nhất:")
            for r in st.recent:
                lines.append(f"ID {r.track_id} {CLASS_LABELS.get(r.class_name, r.class_name)} "
                             f"{DIRECTION_LABELS.get(r.direction, r.direction)}: "
                             f"{r.speed_kmh:.0f} km/h ({r.transit_s:.1f} s, {r.distance_m:.0f} m)")
        self._set_log("\n".join(lines))