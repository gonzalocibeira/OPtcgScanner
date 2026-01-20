#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

# Apple Vision OCR (macOS)
import Quartz
import Vision


# ----------------------------
# Config: fixed ROI (relative)
# ----------------------------
# ROI is a rectangle where you will place the card code area (bottom-right of the card).
# Format: (x, y, w, h) as fractions of frame width/height.
# Tweak these once you see the preview.
ROI_REL = (0.58, 0.68, 0.40, 0.30)

# OCR stability helpers
MIN_CODE_LEN = 7  # e.g., OP05-043 is 8, but keep small guard


# Match common One Piece TCG code formats:
# OPxx-xxx, EBxx-xxx, STxx-xxx, PRBxx-xxx, P-xxx (promos)
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
    # If file is invalid, don’t destroy it; start fresh in-memory.
    return []


def np_bgr_to_cgimage(bgr: np.ndarray):
    """
    Convert a BGR OpenCV image to a CGImage for Vision.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    bytes_per_row = w * 3

    data_provider = Quartz.CGDataProviderCreateWithData(
        None,
        rgb.tobytes(),
        len(rgb.tobytes()),
        None
    )

    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    bitmap_info = Quartz.kCGBitmapByteOrderDefault

    cgimage = Quartz.CGImageCreate(
        w, h,                 # width, height
        8,                    # bits per component
        24,                   # bits per pixel
        bytes_per_row,
        color_space,
        bitmap_info,
        data_provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault
    )
    return cgimage


def vision_ocr_text(bgr_roi: np.ndarray) -> str:
    """
    Run Apple Vision text recognition on an image ROI (BGR ndarray).
    Returns raw recognized text (joined lines).
    """
    cgimage = np_bgr_to_cgimage(bgr_roi)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    # Accuracy vs speed: accurate is better for small card text
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    # If your OS supports it, you can hint languages; English is fine for alphanumerics.

    ok, err = handler.performRequests_error_([request], None)
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
    """
    Extract first valid code from OCR text using regex.
    Also normalizes common OCR confusion (O/0, I/1) lightly.
    """
    if not text or len(text) < MIN_CODE_LEN:
        return None

    # Normalize: remove spaces, common separators
    t = text.upper().replace(" ", "").replace("_", "-")

    # Some OCRs read OP as 0P or O P; do light fixes
    t = t.replace("0P", "OP").replace("O P", "OP").replace("OP-", "OP")
    t = t.replace("EB-", "EB").replace("ST-", "ST").replace("PRB-", "PRB")

    m = CODE_REGEX.search(t)
    return m.group(1) if m else None


def compute_roi(frame_w: int, frame_h: int):
    rx, ry, rw, rh = ROI_REL
    x = int(frame_w * rx)
    y = int(frame_h * ry)
    w = int(frame_w * rw)
    h = int(frame_h * rh)
    # clamp
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess ROI to improve OCR:
    - grayscale
    - resize up
    - contrast
    - adaptive threshold
    Then return as BGR (Vision accepts any image, but this helps).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale (helps small text)
    scale = 2.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive threshold (robust to lighting)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Convert back to BGR
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2...)")
    ap.add_argument("--out", type=str, default="cards.json", help="Output JSON path")
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height")
    args = ap.parse_args()

    out_path = Path(args.out)
    db = load_db(out_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}. Try --camera 1 or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = "OP TCG Scanner (Space/Enter=Scan, D=Undo, Q/Esc=Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    flash_until = 0.0
    flash_text = ""
    flash_ok = False

    last_scan_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        x, y, rw, rh = compute_roi(w, h)

        # Draw guide ROI
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Place card code here",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
            msg = flash_text
            color = (0, 255, 0) if flash_ok else (0, 0, 255)
            cv2.putText(
                frame,
                msg,
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
                (20, 140),
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

        if key in (ord("d"),):  # undo last
            if db:
                removed = db.pop()
                atomic_write_json(out_path, db)
                flash_text = f"UNDO: {removed}"
                flash_ok = True
                flash_until = time.time() + 1.0

        if key in (13, 32):  # Enter or Space = scan
            # Simple debounce
            if time.time() - last_scan_time < 0.25:
                continue
            last_scan_time = time.time()

            roi = frame[y:y + rh, x:x + rw].copy()
            pre = preprocess_for_ocr(roi)

            try:
                text = vision_ocr_text(pre)
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
                flash_text = "No code detected"
                flash_ok = False
                flash_until = time.time() + 1.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()