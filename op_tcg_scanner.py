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
  python -m pip install opencv-python pyobjc pyobjc-framework-Vision pyobjc-framework-Quartz

Run:
  python op_tcg_scanner.py --camera 0 --out cards.json
  (try --camera 1,2... if using iPhone Continuity Camera)
"""

import argparse
import json
import os
import re
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
ROI_REL = (0.60, 0.58, 0.36, 0.22)  # (x, y, w, h) in fractions of frame


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


def compute_roi(frame_w: int, frame_h: int):
    rx, ry, rw, rh = ROI_REL
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2...)")
    ap.add_argument("--out", type=str, default="cards.json", help="Output JSON path")
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height")
    ap.add_argument("--preprocess", action="store_true", help="Enable mild preprocessing before OCR")
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

        # ROI guide box
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Put card code inside this box",
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()