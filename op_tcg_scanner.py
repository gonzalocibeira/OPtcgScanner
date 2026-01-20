#!/usr/bin/env python3
"""
OP TCG Card Code Scanner (macOS)
- OpenCV camera preview + fixed ROI guide box
- Press Space/Enter to scan ROI with Apple Vision OCR
- Saves ONLY card codes to JSON (list of strings)
- Shows SCAN OK + code for 1 second

Python: 3.10+ (recommended 3.11/3.12)
Install:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install opencv-python pyobjc pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-AVFoundation

Run:
  python op_tcg_scanner.py --camera 0 --out cards.json
  (try --camera 1,2... if using iPhone Continuity Camera)
  python op_tcg_scanner.py --backend avfoundation --focus-mode continuous --zoom 1.5
"""

import argparse
import ctypes
import importlib
import importlib.util
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import Quartz
import Vision


# ----------------------------
# Fixed ROI (relative to frame)
# ----------------------------
# Put the card's bottom-right corner (with the code) inside this box.
# Tuned vs your screenshot (moves ROI up and makes it smaller).
ROI_REL = (0.66, 0.62, 0.26, 0.16)  # (x, y, w, h) in fractions of frame


# Match common One Piece TCG code formats (add more if you need)
CODE_REGEX = re.compile(
    r"\b("
    r"OP\d{2}-\d{3}"
    r"|EB\d{2}-\d{3}"
    r"|ST\d{2}-\d{3}"
    r"|PRB\d{2}-\d{3}"
    r"|P-\d{3}"
    r")\b"
)


def atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_db(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    return []


def parse_roi_rel(value: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h with 4 comma-separated values.")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI fractions must be numeric.") from exc
    if not all(math.isfinite(n) for n in nums):
        raise argparse.ArgumentTypeError("ROI fractions must be finite numbers.")
    x, y, w, h = nums
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        raise argparse.ArgumentTypeError("ROI fractions must be within 0-1, with w/h > 0.")
    return x, y, w, h


def parse_roi_px(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI pixels must be x,y,w,h with 4 comma-separated values.")
    try:
        nums = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI pixels must be integers.") from exc
    x, y, w, h = nums
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI pixels must be >= 0 with w/h > 0.")
    return x, y, w, h


def compute_roi(
    frame_w: int,
    frame_h: int,
    roi_rel: tuple[float, float, float, float] | None = None,
    roi_px: tuple[int, int, int, int] | None = None,
):
    if roi_px is not None:
        x, y, w, h = roi_px
    else:
        rx, ry, rw, rh = roi_rel if roi_rel is not None else ROI_REL
        x = int(frame_w * rx)
        y = int(frame_h * ry)
        w = int(frame_w * rw)
        h = int(frame_h * rh)

    x = max(0, min(x, frame_w - 2))
    y = max(0, min(y, frame_h - 2))
    w = max(2, min(w, frame_w - x))
    h = max(2, min(h, frame_h - y))
    return x, y, w, h


def np_bgr_to_cgimage(bgr: np.ndarray):
    """
    Vision is happiest with BGRA 32bpp + proper bitmap flags.
    """
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    bytes_per_row = w * 4
    data = bgra.tobytes()

    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    bitmap_info = (Quartz.kCGBitmapByteOrder32Little |
                   Quartz.kCGImageAlphaPremultipliedFirst)

    cgimage = Quartz.CGImageCreate(
        w, h,
        8, 32,
        bytes_per_row,
        cs,
        bitmap_info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault
    )
    return cgimage


def vision_ocr_text(bgr_img: np.ndarray) -> str:
    cgimage = np_bgr_to_cgimage(bgr_img)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)

    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return ""

    results = request.results() or []
    lines = []
    for obs in results:
        top = obs.topCandidates_(1)
        if top and len(top) > 0:
            s = str(top[0].string())
            if s:
                lines.append(s)
    return "\n".join(lines).strip()


def extract_code(text: str) -> str | None:
    if not text:
        return None

    t = text.upper()

    # Remove spaces and normalize separators (OCR often injects spaces)
    t = t.replace(" ", "").replace("_", "-")

    # Common OCR fixes
    t = t.replace("0P", "OP")  # 0P -> OP
    t = t.replace("OP—", "OP-").replace("OP–", "OP-").replace("OP−", "OP-")  # weird dashes
    t = t.replace("EB—", "EB-").replace("ST—", "ST-").replace("PRB—", "PRB-")

    m = CODE_REGEX.search(t)
    return m.group(1) if m else None


def mild_preprocess(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Mild preprocessing (safe for Vision): upscale + light contrast.
    Avoid heavy thresholding unless you really need it.
    """
    scale = 2.0
    up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


class OpenCVCapture:
    def __init__(self, camera_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {camera_index}. Try --camera 1 or 2.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def load_avfoundation_modules():
    modules = {}
    for name in ("AVFoundation", "CoreMedia", "CoreVideo", "Foundation", "dispatch", "objc"):
        if importlib.util.find_spec(name) is None:
            return None
        modules[name] = importlib.import_module(name)
    return modules


def build_avfoundation_delegate(modules):
    Foundation = modules["Foundation"]
    CoreMedia = modules["CoreMedia"]
    CoreVideo = modules["CoreVideo"]
    objc = modules["objc"]

    class AVFoundationFrameDelegate(Foundation.NSObject):
        def init(self):
            self = objc.super(AVFoundationFrameDelegate, self).init()
            if self is None:
                return None
            self.latest_frame = None
            self.lock = threading.Lock()
            self.frame_event = threading.Event()
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, _output, sampleBuffer, _connection
        ):
            image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
            if image_buffer is None:
                return
            CoreVideo.CVPixelBufferLockBaseAddress(image_buffer, 0)
            try:
                width = CoreVideo.CVPixelBufferGetWidth(image_buffer)
                height = CoreVideo.CVPixelBufferGetHeight(image_buffer)
                bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(image_buffer)
                base_addr = CoreVideo.CVPixelBufferGetBaseAddress(image_buffer)
                if base_addr is None:
                    return
                data = ctypes.string_at(base_addr, bytes_per_row * height)
                frame = np.frombuffer(data, dtype=np.uint8)
                frame = frame.reshape((height, bytes_per_row // 4, 4))
                frame = frame[:, :width, :]
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.latest_frame = bgr
                    self.frame_event.set()
            finally:
                CoreVideo.CVPixelBufferUnlockBaseAddress(image_buffer, 0)

    return AVFoundationFrameDelegate


class AVFoundationCapture:
    def __init__(self, camera_index: int, width: int, height: int, focus_mode: str, zoom: float):
        modules = load_avfoundation_modules()
        if modules is None:
            raise RuntimeError("AVFoundation modules unavailable")

        self.modules = modules
        AVFoundation = modules["AVFoundation"]
        CoreVideo = modules["CoreVideo"]
        dispatch = modules["dispatch"]

        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        if not devices or camera_index >= len(devices):
            raise RuntimeError(f"No AVFoundation camera for index {camera_index}")
        self.device = devices[camera_index]

        self.session = AVFoundation.AVCaptureSession.alloc().init()
        if width >= 1280 or height >= 720:
            self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720)
        else:
            self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset640x480)

        input_device, err = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(
            self.device, None
        )
        if err is not None:
            raise RuntimeError(f"Failed to create AVFoundation input: {err}")
        if self.session.canAddInput_(input_device):
            self.session.addInput_(input_device)

        self.output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        settings = {
            CoreVideo.kCVPixelBufferPixelFormatTypeKey: CoreVideo.kCVPixelFormatType_32BGRA
        }
        self.output.setVideoSettings_(settings)
        self.output.setAlwaysDiscardsLateVideoFrames_(True)

        delegate_cls = build_avfoundation_delegate(modules)
        self.delegate = delegate_cls.alloc().init()
        self.queue = dispatch.dispatch_queue_create(
            "op_tcg_scanner.avfoundation", dispatch.DISPATCH_QUEUE_SERIAL
        )
        self.output.setSampleBufferDelegate_queue_(self.delegate, self.queue)
        if self.session.canAddOutput_(self.output):
            self.session.addOutput_(self.output)

        self.configure_device(focus_mode, zoom)
        self.session.startRunning()

    def configure_device(self, focus_mode: str, zoom: float):
        AVFoundation = self.modules["AVFoundation"]
        if not self.device.lockForConfiguration_(None):
            return
        try:
            focus_map = {
                "auto": AVFoundation.AVCaptureFocusModeAutoFocus,
                "continuous": AVFoundation.AVCaptureFocusModeContinuousAutoFocus,
                "locked": AVFoundation.AVCaptureFocusModeLocked,
            }
            mode = focus_map.get(focus_mode)
            if mode is not None and self.device.isFocusModeSupported_(mode):
                self.device.setFocusMode_(mode)

            if zoom and zoom > 1.0:
                max_zoom = float(self.device.activeFormat().videoMaxZoomFactor())
                clamped = max(1.0, min(float(zoom), max_zoom))
                self.device.setVideoZoomFactor_(clamped)
        finally:
            self.device.unlockForConfiguration()

    def read(self):
        if not self.delegate.frame_event.wait(1.0):
            return False, None
        with self.delegate.lock:
            frame = None if self.delegate.latest_frame is None else self.delegate.latest_frame.copy()
            self.delegate.frame_event.clear()
        return (frame is not None), frame

    def release(self):
        self.session.stopRunning()


def create_capture(args):
    if args.backend == "avfoundation":
        try:
            return AVFoundationCapture(
                args.camera, args.width, args.height, args.focus_mode, args.zoom
            )
        except RuntimeError as exc:
            print(f"AVFoundation backend unavailable: {exc}. Falling back to OpenCV.")
    return OpenCVCapture(args.camera, args.width, args.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2...)")
    ap.add_argument("--out", type=str, default="cards.json", help="Output JSON path")
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height")
    ap.add_argument(
        "--backend",
        choices=("opencv", "avfoundation"),
        default="opencv",
        help="Capture backend (avfoundation uses PyObjC on macOS)",
    )
    ap.add_argument(
        "--focus-mode",
        choices=("none", "auto", "continuous", "locked"),
        default="none",
        help="AVFoundation focus mode",
    )
    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="AVFoundation zoom factor (1.0 = no zoom)",
    )
    ap.add_argument(
        "--roi",
        type=parse_roi_rel,
        help="ROI as fractions x,y,w,h (0-1). Overrides default guide box.",
    )
    ap.add_argument(
        "--roi-px",
        type=parse_roi_px,
        help="ROI in pixels x,y,w,h. Overrides --roi and default guide box.",
    )
    ap.add_argument("--preprocess", action="store_true", help="Enable mild preprocessing before OCR")
    args = ap.parse_args()

    out_path = Path(args.out)
    db = load_db(out_path)

    capture = create_capture(args)

    win = "OP TCG Scanner (Space/Enter=Scan, D=Undo, Q/Esc=Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    flash_until = 0.0
    flash_text = ""
    flash_ok = False

    last_scan_time = 0.0

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        x, y, rw, rh = compute_roi(w, h, args.roi, args.roi_px)

        # ROI guide box
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Put card code inside this small box",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Stats
        cv2.putText(
            frame,
            f"Scanned: {len(db)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Flash overlay
        now = time.time()
        if now < flash_until:
            color = (0, 255, 0) if flash_ok else (0, 0, 255)
            cv2.putText(
                frame,
                flash_text,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "SCAN OK" if flash_ok else "SCAN FAIL",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):  # Esc or q
            break

        if key == ord("d"):  # undo last
            if db:
                removed = db.pop()
                atomic_write_json(out_path, db)
                flash_text = f"UNDO: {removed}"
                flash_ok = True
                flash_until = time.time() + 1.0

        if key in (13, 32):  # Enter or Space
            if time.time() - last_scan_time < 0.25:
                continue
            last_scan_time = time.time()

            roi = frame[y:y + rh, x:x + rw].copy()
            img_for_ocr = mild_preprocess(roi) if args.preprocess else roi

            try:
                text = vision_ocr_text(img_for_ocr)
                code = extract_code(text)
            except Exception:
                code = None

            if code:
                db.append(code)
                atomic_write_json(out_path, db)
                flash_text = code
                flash_ok = True
                flash_until = time.time() + 1.0
            else:
                # Useful hint: show what Vision saw (short)
                short = (text.replace("\n", " ")[:40] + "...") if text else "No text"
                flash_text = f"No code ({short})"
                flash_ok = False
                flash_until = time.time() + 1.0

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
