#!/usr/bin/env python3
"""Generate a small synthetic multi-page TIFF for local demos.

Example:
    python tools/generate_demo_tiff.py assets/demo.tiff --pages 3 --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_tiff(path: Path, pages: int = 2, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(pages):
        img = Image.new("RGB", (1200, 300), color="white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = f"DEMO OCR PAGE {i + 1}\n\nThis is a synthetic TIFF for local pipeline demos."
        draw.text((40, 40), text, fill="black", font=font)
        images.append(img)

    # Save as multi-page TIFF
    first, rest = images[0], images[1:]
    first.save(
        path,
        format="TIFF",
        dpi=(dpi, dpi),
        save_all=True,
        append_images=rest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a demo multi-page TIFF")
    parser.add_argument("output", nargs="?", default="assets/demo.tiff")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    out = Path(args.output)
    generate_tiff(out, pages=args.pages, dpi=args.dpi)
    print(f"Written demo TIFF: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
