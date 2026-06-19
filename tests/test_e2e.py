"""
End-to-end pipeline tests.

These tests exercise the full path from "TIFF file on disk" → DB registration
→ process_image → mark_complete → verified output files and DB state.

Two variants:
  1. mock_ocr  — always runs; uses MockOCREngine (no Tesseract needed)
  2. real_ocr  — skipped when the `tesseract` binary is not on PATH;
                 uses a PIL-drawn TIFF and asserts that recognised text
                 contains the expected string
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from pipeline.config import (
    Config,
    DatabaseConfig,
    InputConfig,
    OCRConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from pipeline.db import Base, OCRImage
from pipeline.preprocessor import ImagePreprocessor
from pipeline.worker import mark_complete, process_image
from tests.conftest import MockOCREngine


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def e2e_dirs(tmp_path):
    dirs = {
        "input": tmp_path / "input",
        "output": tmp_path / "output",
        "logs": tmp_path / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture()
def e2e_cfg(e2e_dirs):
    return Config(
        database=DatabaseConfig(),
        input=InputConfig(
            root_path=str(e2e_dirs["input"]),
            extensions=[".tif", ".tiff"],
            recursive=True,
        ),
        output=OutputConfig(
            root_path=str(e2e_dirs["output"]),
            formats=["txt", "json"],
        ),
        ocr=OCRConfig(engine="tesseract", language="eng"),
        preprocessing=PreprocessingConfig(
            enabled=True,
            min_dpi=200,
            target_dpi=300,
            default_dpi=300,
            deskew=False,  # keep tests fast
            denoise=False,
            binarization="otsu",
        ),
        pipeline=PipelineConfig(
            num_workers=1,
            batch_size=5,
            max_retries=3,
            log_dir=str(e2e_dirs["logs"]),
        ),
    )


@pytest.fixture()
def e2e_engine():
    from sqlalchemy import create_engine

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _insert_pending(engine, file_path: str) -> int:
    """Insert an OCRImage row with status='pending' and return its id."""
    p = Path(file_path)
    with Session(engine) as s:
        img = OCRImage(
            file_path=file_path,
            file_name=p.name,
            file_size_bytes=p.stat().st_size,
            status="pending",
            retry_count=0,
        )
        s.add(img)
        s.commit()
        return img.id


# ── E2E with MockOCREngine ────────────────────────────────────────────────────


def test_e2e_single_page_mock_ocr(e2e_dirs, e2e_cfg, e2e_engine):
    """
    Full pipeline for a single-page TIFF:
      create TIFF → register in DB → process_image → mark_complete →
      assert DB status=complete, OCRResult row present, output files written.
    """
    # 1. Create a synthetic single-page TIFF
    img = Image.new("L", (400, 100), color=255)
    tiff_path = e2e_dirs["input"] / "doc.tiff"
    img.save(str(tiff_path), format="TIFF", dpi=(300, 300))

    # 2. Register in DB
    image_id = _insert_pending(e2e_engine, str(tiff_path))

    # 3. Process
    preprocessor = ImagePreprocessor(e2e_cfg.preprocessing)
    ocr_engine = MockOCREngine(text="Hello OCR World")

    result = process_image(
        file_path=str(tiff_path),
        input_root=str(e2e_dirs["input"]),
        output_root=str(e2e_dirs["output"]),
        cfg=e2e_cfg,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
    )

    # 4. Persist results
    mark_complete(e2e_engine, image_id, result, "mock")

    # 5. Verify DB state
    with Session(e2e_engine) as s:
        img_row = s.get(OCRImage, image_id)
        assert img_row.status == "complete"
        assert img_row.page_count == 1
        assert len(img_row.results) == 1
        assert img_row.results[0].extracted_text == "Hello OCR World"
        assert img_row.results[0].ocr_engine == "mock"

    # 6. Verify output files
    stem = tiff_path.stem
    txt_path = e2e_dirs["output"] / f"{stem}.txt"
    json_path = e2e_dirs["output"] / f"{stem}.json"

    assert txt_path.exists(), "txt output file must be written"
    assert json_path.exists(), "json output file must be written"

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "Hello OCR World" in txt_content

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["page_count"] == 1
    assert "Hello OCR World" in doc["full_text"]
    assert doc["pages"][0]["page_number"] == 1


def test_e2e_multipage_tiff_mock_ocr(e2e_dirs, e2e_cfg, e2e_engine):
    """Multi-page TIFF: every page is OCR'd and one OCRResult row is created per page."""
    pages = [Image.new("L", (400, 100), color=255) for _ in range(3)]
    tiff_path = e2e_dirs["input"] / "multipage.tiff"
    pages[0].save(
        str(tiff_path),
        format="TIFF",
        save_all=True,
        append_images=pages[1:],
        dpi=(300, 300),
    )

    image_id = _insert_pending(e2e_engine, str(tiff_path))

    preprocessor = ImagePreprocessor(e2e_cfg.preprocessing)
    ocr_engine = MockOCREngine(text="Page text")

    result = process_image(
        file_path=str(tiff_path),
        input_root=str(e2e_dirs["input"]),
        output_root=str(e2e_dirs["output"]),
        cfg=e2e_cfg,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
    )
    mark_complete(e2e_engine, image_id, result, "mock")

    with Session(e2e_engine) as s:
        img_row = s.get(OCRImage, image_id)
        assert img_row.page_count == 3
        assert len(img_row.results) == 3
        page_numbers = {r.page_number for r in img_row.results}
        assert page_numbers == {1, 2, 3}

    # JSON output must list all 3 pages
    doc = json.loads((e2e_dirs["output"] / "multipage.json").read_text())
    assert doc["page_count"] == 3
    assert len(doc["pages"]) == 3


def test_e2e_output_directory_mirrors_input_structure(e2e_dirs, e2e_cfg, e2e_engine):
    """Files in sub-directories produce mirrored output paths."""
    sub = e2e_dirs["input"] / "batch01"
    sub.mkdir()
    img = Image.new("L", (200, 50), color=255)
    tiff_path = sub / "scan.tiff"
    img.save(str(tiff_path), format="TIFF", dpi=(300, 300))

    image_id = _insert_pending(e2e_engine, str(tiff_path))

    preprocessor = ImagePreprocessor(e2e_cfg.preprocessing)
    result = process_image(
        file_path=str(tiff_path),
        input_root=str(e2e_dirs["input"]),
        output_root=str(e2e_dirs["output"]),
        cfg=e2e_cfg,
        preprocessor=preprocessor,
        ocr_engine=MockOCREngine(),
    )
    mark_complete(e2e_engine, image_id, result, "mock")

    expected_txt = e2e_dirs["output"] / "batch01" / "scan.txt"
    assert expected_txt.exists(), (
        "output tree must mirror input sub-directory structure"
    )


# ── E2E with real Tesseract (skipped when binary absent) ─────────────────────

_TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


@pytest.mark.skipif(not _TESSERACT_AVAILABLE, reason="tesseract binary not installed")
def test_e2e_real_tesseract_ocr(e2e_dirs, e2e_engine):
    """
    Draw a high-contrast text TIFF and verify Tesseract can read it.

    This is the one test that exercises the real TesseractEngine path end-to-end
    with a live binary, proving the pipeline works for genuine OCR workloads.
    """
    from pipeline.config import OCRConfig
    from pipeline.ocr_engine import TesseractEngine

    expected_word = "HELLO"

    # Build a large, high-contrast TIFF Tesseract can reliably read
    img = Image.new("RGB", (800, 150), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=60)
    draw.text((50, 30), expected_word, fill="black", font=font)

    tiff_path = e2e_dirs["input"] / "real_ocr.tiff"
    img.save(str(tiff_path), format="TIFF", dpi=(300, 300))

    cfg = Config(
        database=DatabaseConfig(),
        input=InputConfig(root_path=str(e2e_dirs["input"])),
        output=OutputConfig(
            root_path=str(e2e_dirs["output"]),
            formats=["txt", "json"],
        ),
        ocr=OCRConfig(
            engine="tesseract", language="eng", tesseract_config="--oem 1 --psm 6"
        ),
        preprocessing=PreprocessingConfig(
            enabled=True,
            min_dpi=200,
            target_dpi=300,
            default_dpi=300,
            deskew=False,
            denoise=False,
            binarization="none",  # pass clean synthetic image straight to Tesseract
        ),
        pipeline=PipelineConfig(log_dir=str(e2e_dirs["logs"])),
    )

    image_id = _insert_pending(e2e_engine, str(tiff_path))

    preprocessor = ImagePreprocessor(cfg.preprocessing)
    ocr_engine = TesseractEngine(cfg.ocr)

    result = process_image(
        file_path=str(tiff_path),
        input_root=str(e2e_dirs["input"]),
        output_root=str(e2e_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
    )
    mark_complete(e2e_engine, image_id, result, "tesseract")

    with Session(e2e_engine) as s:
        img_row = s.get(OCRImage, image_id)
        assert img_row.status == "complete"
        full_text = " ".join(r.extracted_text or "" for r in img_row.results)

    assert expected_word in full_text.upper(), (
        f"Tesseract should have recognised '{expected_word}' in: {full_text!r}"
    )
