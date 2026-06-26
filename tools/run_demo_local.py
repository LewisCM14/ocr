#!/usr/bin/env python3
"""Simple local demo runner that processes a TIFF into text/JSON without DB.

This is intentionally lightweight and does not depend on MSSQL or the full
pipeline database. It uses `pytesseract` if available, otherwise falls back to
placeholder text to demonstrate the output formats and directory layout.

Example:
    python tools/run_demo_local.py assets/demo.tiff --out demo_output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image


def ocr_page_text(img: Image.Image) -> tuple[str, float, int]:
    """Return (text, confidence, processing_time_ms).

    Uses pytesseract when available; otherwise returns placeholder text with
    confidence 0.0 and a processing time of 0.
    """
    try:
        import pytesseract

        t0 = time.monotonic()
        text = pytesseract.image_to_string(img).strip()
        t_ms = int((time.monotonic() - t0) * 1000)
        # pytesseract has no easy aggregated confidence here; return 1.0 as a
        # best-effort placeholder when text is present.
        conf = 1.0 if text else 0.0
        return text, conf, t_ms
    except Exception:
        return "DEMO OCR TEXT (pytesseract not available)", 0.0, 0


def process_tiff(input_tiff: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(input_tiff)

    pages = []
    texts = []
    page_num = 0

    try:
        while True:
            page_num += 1
            frame = img.copy()
            text, conf, t_ms = ocr_page_text(frame)
            pages.append(
                {
                    "page_number": page_num,
                    "text": text,
                    "confidence": conf,
                    "word_count": len(text.split()) if text else 0,
                    "char_count": len(text) if text else 0,
                    "processing_time_ms": t_ms,
                }
            )
            if text:
                texts.append(text)
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    full_text = "\n\n--- Page Break ---\n\n".join(texts)
    base = out_dir / input_tiff.stem
    (base.with_suffix(".txt")).write_text(full_text, encoding="utf-8")

    # Determine engine used (pytesseract when available)
    try:
        engine_name = "pytesseract"
    except Exception:
        engine_name = "mock"

    json_doc = {
        "source_path": str(input_tiff),
        "file_name": input_tiff.name,
        "relative_path": input_tiff.name,
        "file_size_bytes": input_tiff.stat().st_size,
        "page_count": page_num,
        "full_text": full_text,
        "pages": pages,
        "ocr_engine": engine_name,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (base.with_suffix(".json")).write_text(
        json.dumps(json_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote demo outputs to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight OCR demo on a TIFF")
    parser.add_argument("input", nargs="?", default="assets/demo.tiff")
    parser.add_argument("--out", default="demo_output")
    args = parser.parse_args(argv)

    input_tiff = Path(args.input)
    if not input_tiff.exists():
        print(f"Input TIFF not found: {input_tiff}")
        return 2

    process_tiff(input_tiff, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
