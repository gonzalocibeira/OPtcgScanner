# OP TCG Scanner

A macOS-focused camera scanner for One Piece TCG card codes. It uses OpenCV for live video capture and Apple Vision for OCR, then enriches scans by fetching card data from the public OPTCG API.

## Features

- Live camera preview with a fixed ROI guide box for the card code.
- OCR via Apple Vision (macOS) with optional preprocessing.
- Auto-save to JSON, with quantity tracking and undo support.
- Optional debug logging of failed scans.
- Enriched card payloads + card image preview when available.

## Requirements

- **macOS** (Vision/Quartz frameworks are macOS-only)
- **Python 3.10+** (3.11/3.12 recommended)
- **Homebrew** (recommended for installing Python)
- **Network access** to `https://www.optcgapi.com` for card metadata and images

### Python dependencies

Installed via pip:

- `opencv-python`
- `numpy`
- `pyobjc`
- `pyobjc-framework-Vision`
- `pyobjc-framework-Quartz`

## Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install opencv-python numpy pyobjc pyobjc-framework-Vision pyobjc-framework-Quartz
```

> Note: If you already have a `python3` from Homebrew, it will be used by default. You can check with `which python3`.

## Running

```bash
python op_tcg_scanner.py --camera 0 --out cards.json
```

If you have multiple cameras (e.g., Continuity Camera), try other indices:

```bash
python op_tcg_scanner.py --camera 1
python op_tcg_scanner.py --camera 2
```

### Common options

```bash
python op_tcg_scanner.py \
  --camera 0 \
  --out cards.json \
  --width 1280 \
  --height 720 \
  --preprocess \
  --debug \
  --debug-log debug_scans.log
```

### ROI tuning

The ROI is where the card code should appear. You can adjust it in two ways:

- **Relative ROI** (fractions of the frame):
  ```bash
  python op_tcg_scanner.py --roi 0.66,0.62,0.26,0.16
  ```

- **Pixel ROI** (absolute pixels, overrides `--roi`):
  ```bash
  python op_tcg_scanner.py --roi-px 850,500,300,150
  ```

## Controls

- **Space / Enter**: scan ROI
- **Q / Esc**: quit
- **D**: undo last scan
- **Arrow keys**: move ROI
- **+ / -**: adjust ROI width
- **, / .**: adjust ROI height
- **R**: reset ROI to defaults

## Output format

`--out` writes a JSON list. Each entry is:

```json
{
  "code": "OP09-071",
  "card": { "name": "...", "card_image": "..." },
  "quantity": 2
}
```

If the file already exists, entries are merged and quantities incremented.

## macOS execution notes

1. **Camera permissions**: The first run will prompt for camera access. Grant access to your terminal (or IDE) in **System Settings → Privacy & Security → Camera**.
2. **Apple Vision**: OCR uses the built-in macOS Vision framework via `pyobjc`.
3. **Performance**: Use `--preprocess` only if OCR struggles; it adds extra compute.

## Troubleshooting

- **Black/blank preview**: try a different camera index with `--camera`.
- **OCR misses codes**: ensure the code is inside the ROI box; try `--preprocess`.
- **No card metadata**: check network access to `https://www.optcgapi.com`.

## License

See `LICENSE.txt`.
