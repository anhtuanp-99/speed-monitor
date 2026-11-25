"""Kiểm tra môi trường chạy một lượt: python check_env.py

In ra phiên bản Python / PyTorch / GPU, rồi thử nạp từng file native của torchvision để
hiện LỖI THẬT (torchvision thường nuốt lỗi này và chỉ báo
"operator torchvision::nms does not exist"). Với mỗi file còn in kích thước và mã băm
SHA256 để so sánh giữa hai venv (cùng phiên bản mà mã băm khác nhau = file bị hỏng/khác bản build).
"""
import glob
import hashlib
import importlib
import importlib.util
import os
import platform
import sys

WINERROR_HINTS = {
    126: "thiếu một DLL mà file này phụ thuộc (thường là Microsoft Visual C++ Redistributable 2015-2022 x64)",
    193: "file hỏng, tải dở dang, hoặc sai kiến trúc (không phải x64)",
    225: "phần mềm diệt virus (Windows Defender...) đã chặn hoặc cách ly file",
    1260: "Windows chặn file bằng chính sách (Application Control / Smart App Control / AppLocker)",
    5: "không có quyền truy cập file",
}


def step(title):
    print(f"\n== {title}")


def explain(exc):
    """Trả về chuỗi các lỗi lồng nhau (lỗi ngoài cùng -> lỗi gốc) và mã WinError nếu có."""
    lines, codes, seen = [], [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, "winerror", None)
        text = f"{type(exc).__name__}: {exc}"
        if code is not None:
            text += f"  [WinError {code}]"
            codes.append(code)
        lines.append(text)
        exc = exc.__cause__ or exc.__context__
    return lines, codes


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    print("Python:", sys.version.split()[0], "|", platform.platform())

    step("PyTorch")
    try:
        import torch
    except Exception as exc:                       # noqa: BLE001
        print("LỖI khi import torch:", repr(exc))
        return 1
    print("torch", torch.__version__, "| CUDA khả dụng:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("-> PyTorch không thấy GPU (bản +cpu, hoặc thiếu driver NVIDIA).")

    step("File native của torchvision")
    all_codes = []
    spec = importlib.util.find_spec("torchvision")
    if spec is None or not spec.origin:
        print("Không tìm thấy torchvision trong venv này.")
    else:
        folder = os.path.dirname(spec.origin)
        ext = "*.pyd" if os.name == "nt" else "*.so"
        files = sorted(glob.glob(os.path.join(folder, ext)))
        if not files:
            print(f"Không có file {ext} nào trong {folder}: bản cài torchvision bị thiếu hoặc hỏng.")
        for path in files:
            name = os.path.basename(path)
            info = f"{os.path.getsize(path):,} byte, sha256 {sha256_of(path)[:16]}..."
            try:
                torch.ops.load_library(path)
                print(f"  nạp được  {name}  ({info})")
            except Exception as exc:               # noqa: BLE001
                print(f"  KHÔNG NẠP ĐƯỢC  {name}  ({info})")
                lines, codes = explain(exc)
                all_codes += codes
                for depth, line in enumerate(lines):
                    print("    " + "   " * depth + ("-> " if depth == 0 else "gây ra bởi: ") + line)

    step("torchvision")
    try:
        import torchvision
        from torchvision.ops import nms            # noqa: F401
        print("torchvision", torchvision.__version__, "| nms: OK")
    except Exception as exc:                       # noqa: BLE001
        print("LỖI:", repr(exc))

    step("OpenCV")
    try:
        import cv2
        print("opencv", cv2.__version__)
    except Exception as exc:                       # noqa: BLE001
        print("LỖI:", repr(exc))

    step("Ultralytics")
    try:
        ultralytics = importlib.import_module("ultralytics")
        print("ultralytics", ultralytics.__version__)
    except Exception as exc:                       # noqa: BLE001
        print("LỖI:", repr(exc))

    if all_codes:
        step("Gợi ý theo mã lỗi")
        for code in sorted(set(all_codes)):
            print(f"  WinError {code}: {WINERROR_HINTS.get(code, 'xem tài liệu mã lỗi Windows')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())