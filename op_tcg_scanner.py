#!/usr/bin/env python3
"""
OP TCG Card Code Scanner (macOS)
- OpenCV camera preview + fixed ROI guide box
- Press Space/Enter to scan ROI with Apple Vision OCR
- Saves card codes + card payloads to JSON (list of dicts)
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
  python op_tcg_scanner.py --debug --debug-log debug_scans.log
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from urllib import request

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


# Match common One Piece TCG code formats (add more if you need).
# Tolerate suffix letters/digits (e.g., OP09-071A, OP09-077C3) but strip them from the match.
CODE_REGEX = re.compile(
    r"(?<![A-Z0-9])("
    r"OP\d{2}-\d{3}"
    r"|EB\d{2}-\d{3}"
    r"|ST\d{2}-\d{3}"
    r"|PRB\d{2}-\d{3}"
    r"|P-\d{3}"
    r")(?=(?:[A-Z0-9]{1,3})?[^A-Z0-9]|$)"
)


def atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_debug_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def load_db(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if all(isinstance(x, dict) and "code" in x for x in data):
                normalized = []
                for item in data:
                    quantity = item.get("quantity", 1)
                    try:
                        quantity = int(quantity)
                    except (TypeError, ValueError):
                        quantity = 1
                    if quantity < 1:
                        quantity = 1
                    normalized.append(
                        {
                            "code": item.get("code"),
                            "card": item.get("card"),
                            "quantity": quantity,
                        }
                    )
                return normalized
            if all(isinstance(x, str) for x in data):
                return [{"code": code, "card": None, "quantity": 1} for code in data]
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
    normalized = normalize_text(text)
    return extract_code_from_normalized(normalized)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.upper()

    # Normalize dash variants globally before any targeted fixes.
    normalized = re.sub(r"[‐‑‒–—―−]", "-", normalized)
    # Replace separator noise (spaces, underscores, punctuation) with dashes.
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"[^A-Z0-9-]+", "-", normalized)
    # Insert missing dashes before set prefixes when glued to prior alphanumerics.
    normalized = re.sub(r"([A-Z0-9])(OP|EB|ST|PRB|P-)", r"\1-\2", normalized)

    # Common OCR fixes after dash normalization.
    normalized = normalized.replace("0P", "OP")  # 0P -> OP
    return normalized


def extract_code_from_normalized(normalized: str) -> str | None:
    if not normalized:
        return None

    m = CODE_REGEX.search(normalized)
    if m:
        return m.group(1)

    candidate_regex = re.compile(
        r"(?:(OP|EB|ST|PRB)([0-9OISBLD]{2})-([0-9OISBLD]{3})|P-([0-9OISBLD]{3}))"
    )
    correction_map = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "D": "0"})

    for match in candidate_regex.finditer(normalized):
        if match.group(1):
            set_prefix, set_num, card_num = match.group(1), match.group(2), match.group(3)
            corrected = f"{set_prefix}{set_num.translate(correction_map)}-{card_num.translate(correction_map)}"
        else:
            corrected = f"P-{match.group(4).translate(correction_map)}"

        strict_match = CODE_REGEX.search(corrected)
        if strict_match:
            return strict_match.group(1)

    return None


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


def fetch_card_info(code: str) -> dict | None:
    card_set_id = code.strip()
    if not card_set_id:
        return None

    url = f"https://www.optcgapi.com/api/sets/card/{card_set_id}/"
    try:
        with request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except Exception:
        return None

    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict):
        return payload
    return None


def get_text_size(text: str, font, scale: float, thickness: int) -> tuple[int, int, int]:
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    return text_w, text_h, baseline


def clamp_text_origin(
    origin_x: int,
    origin_y: int,
    text_w: int,
    text_h: int,
    frame_w: int,
    frame_h: int,
    padding: int = 2,
) -> tuple[int, int]:
    min_x = padding
    max_x = max(padding, frame_w - text_w - padding)
    min_y = text_h + padding
    max_y = max(min_y, frame_h - padding)
    clamped_x = max(min_x, min(origin_x, max_x))
    clamped_y = max(min_y, min(origin_y, max_y))
    return clamped_x, clamped_y


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


def main():
    ap = argparse.ArgumentParser(
        description="OP TCG card code scanner using OpenCV capture + Apple Vision OCR."
    )
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2...)")
    ap.add_argument("--out", type=str, default="cards.json", help="Output JSON path")
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height")
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
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for failed scans (no valid code found).",
    )
    ap.add_argument(
        "--debug-log",
        type=str,
        default="debug_scans.log",
        help="Debug log path (used with --debug).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    db = load_db(out_path)
    db_by_code = {
        entry["code"]: entry
        for entry in db
        if isinstance(entry, dict) and entry.get("code")
    }
    scan_history = []
    debug_log_path = Path(args.debug_log)

    capture = OpenCVCapture(args.camera, args.width, args.height)

    win = ("OP TCG Scanner (Space/Enter=Scan, D=Undo, Q/Esc=Quit, "
           "ROI: Arrows/WASD move, +/- width, ,/. height, R reset)")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    flash_until = 0.0
    flash_text = ""
    flash_ok = False

    last_scan_time = 0.0
    roi_rel_state = args.roi if args.roi is not None else ROI_REL
    roi_px_pending = args.roi_px
    roi_initialized = roi_px_pending is None

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        base_scale = max(0.6, min(1.0, min(w / 1280, h / 720)))
        if not roi_initialized and roi_px_pending is not None:
            rx, ry, rw, rh = compute_roi(w, h, roi_px=roi_px_pending)
            roi_rel_state = (rx / w, ry / h, rw / w, rh / h)
            roi_initialized = True

        x, y, rw, rh = compute_roi(w, h, roi_rel=roi_rel_state)

        # ROI guide box
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        hint_text = "Put card code inside this small box"
        hint_font = cv2.FONT_HERSHEY_SIMPLEX
        hint_scale = max(0.5, min(0.9, base_scale * 0.85))
        hint_thickness = 2
        hint_pad = 6
        hint_w, hint_h, _ = get_text_size(hint_text, hint_font, hint_scale, hint_thickness)
        if hint_w + hint_pad * 2 <= rw and hint_h + hint_pad * 2 <= rh:
            hint_origin = (x + hint_pad, y + hint_pad + hint_h)
        else:
            if y - hint_pad - hint_h >= 0:
                hint_origin = (x, y - hint_pad)
            else:
                hint_origin = (x, y + rh + hint_pad + hint_h)
        hint_origin = clamp_text_origin(
            hint_origin[0],
            hint_origin[1],
            hint_w,
            hint_h,
            w,
            h,
            padding=hint_pad,
        )
        cv2.putText(
            frame,
            hint_text,
            hint_origin,
            hint_font,
            hint_scale,
            (255, 255, 255),
            hint_thickness,
            cv2.LINE_AA
        )

        # Stats
        stats_scale = max(0.7, base_scale)
        total_quantity = sum(
            entry.get("quantity", 1) for entry in db if isinstance(entry, dict)
        )
        cv2.putText(
            frame,
            f"Scanned (total): {total_quantity}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            stats_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        hud_font = cv2.FONT_HERSHEY_SIMPLEX
        if h < 520:
            hud_scale = max(0.45, base_scale * 0.55)
            hud_line = "ROI: Move=Arrows  Width=+/-  Height=,/.  R=Reset ROI"
            hud_w, hud_h, _ = get_text_size(hud_line, hud_font, hud_scale, 2)
            hud_origin = clamp_text_origin(20, h - 20, hud_w, hud_h, w, h, padding=10)
            cv2.putText(
                frame,
                hud_line,
                hud_origin,
                hud_font,
                hud_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        else:
            hud_scale = max(0.6, base_scale * 0.7)
            hud_line_1 = "ROI: Move=Arrows  Width=+/-  Height=,/."
            hud_line_2 = "R=Reset ROI"
            hud_w1, hud_h1, _ = get_text_size(hud_line_1, hud_font, hud_scale, 2)
            hud_w2, hud_h2, _ = get_text_size(hud_line_2, hud_font, hud_scale, 2)
            hud_origin_1 = clamp_text_origin(20, h - 60, hud_w1, hud_h1, w, h, padding=10)
            hud_origin_2 = clamp_text_origin(20, h - 30, hud_w2, hud_h2, w, h, padding=10)
            cv2.putText(
                frame,
                hud_line_1,
                hud_origin_1,
                hud_font,
                hud_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                hud_line_2,
                hud_origin_2,
                hud_font,
                hud_scale,
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
            if scan_history:
                removed_code = scan_history.pop()
                entry = db_by_code.get(removed_code)
                if entry:
                    quantity = entry.get("quantity", 1)
                    if quantity > 1:
                        entry["quantity"] = quantity - 1
                    else:
                        db.remove(entry)
                        db_by_code.pop(removed_code, None)
                    atomic_write_json(out_path, db)
                    flash_text = f"UNDO: {removed_code}"
                    flash_ok = True
                    flash_until = time.time() + 1.0

        move_step = max(2, int(min(w, h) * 0.01))
        size_step = max(2, int(min(w, h) * 0.005))
        move_map = {
            81: (-1, 0),  # Left arrow
            82: (0, -1),  # Up arrow
            83: (1, 0),   # Right arrow
            84: (0, 1),   # Down arrow
        }

        if key in move_map:
            dx, dy = move_map[key]
            nx = x + dx * move_step
            ny = y + dy * move_step
            nx, ny, nrw, nrh = compute_roi(w, h, roi_px=(nx, ny, rw, rh))
            roi_rel_state = (nx / w, ny / h, nrw / w, nrh / h)

        if key in (ord("+"), ord("=")):
            nx, ny, nrw, nrh = compute_roi(w, h, roi_px=(x, y, rw + size_step, rh))
            roi_rel_state = (nx / w, ny / h, nrw / w, nrh / h)
        if key in (ord("-"), ord("_")):
            nx, ny, nrw, nrh = compute_roi(w, h, roi_px=(x, y, rw - size_step, rh))
            roi_rel_state = (nx / w, ny / h, nrw / w, nrh / h)
        if key == ord("."):
            nx, ny, nrw, nrh = compute_roi(w, h, roi_px=(x, y, rw, rh + size_step))
            roi_rel_state = (nx / w, ny / h, nrw / w, nrh / h)
        if key == ord(","):
            nx, ny, nrw, nrh = compute_roi(w, h, roi_px=(x, y, rw, rh - size_step))
            roi_rel_state = (nx / w, ny / h, nrw / w, nrh / h)
        if key == ord("r"):
            roi_rel_state = ROI_REL

        if key in (13, 32):  # Enter or Space
            if time.time() - last_scan_time < 0.25:
                continue
            last_scan_time = time.time()

            roi = frame[y:y + rh, x:x + rw].copy()
            img_for_ocr = mild_preprocess(roi) if args.preprocess else roi

            text = ""
            normalized_text = ""
            code = None
            error_message = None
            try:
                text = vision_ocr_text(img_for_ocr)
                normalized_text = normalize_text(text)
                code = extract_code_from_normalized(normalized_text)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"

            if code:
                card_payload = fetch_card_info(code)
                entry = db_by_code.get(code)
                if entry:
                    entry["quantity"] = entry.get("quantity", 1) + 1
                    if entry.get("card") is None and card_payload is not None:
                        entry["card"] = card_payload
                else:
                    entry = {"code": code, "card": card_payload, "quantity": 1}
                    db.append(entry)
                    db_by_code[code] = entry
                scan_history.append(code)
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
                if args.debug:
                    append_debug_log(
                        debug_log_path,
                        {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "ocr_text": text,
                            "normalized_text": normalized_text,
                            "regex": CODE_REGEX.pattern,
                            "roi_px": [x, y, rw, rh],
                            "roi_rel": [roi_rel_state[0], roi_rel_state[1], roi_rel_state[2], roi_rel_state[3]],
                            "preprocess": args.preprocess,
                            "error": error_message,
                        },
                    )

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
