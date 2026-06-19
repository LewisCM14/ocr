"""
Shared pytest fixtures for the OCR pipeline test suite.

All tests use:
  - SQLite in-memory databases (no MSSQL driver required)
  - Synthetic PIL-generated TIFF images (no external files)
  - A MockOCREngine that returns deterministic text (no Tesseract required
    for unit tests; the E2E module provides a real-Tesseract variant)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine

# Ensure project root is importable when running without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    Config,
    DatabaseConfig,
    InputConfig,
    OCRConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from pipeline.db import Base
from pipeline.ocr_engine import PageOCRResult


# ── Database ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sqlite_engine():
    """Fresh in-memory SQLite engine with all pipeline tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


# ── Directory layout ──────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dirs(tmp_path):
    """Standard temp directory tree: input/, output/, logs/."""
    dirs = {
        "root": tmp_path,
        "input": tmp_path / "input",
        "output": tmp_path / "output",
        "logs": tmp_path / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Config ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def base_cfg(tmp_dirs):
    """A Config that points everything at tmp_dirs."""
    return Config(
        database=DatabaseConfig(),
        input=InputConfig(
            root_path=str(tmp_dirs["input"]),
            extensions=[".tif", ".tiff"],
            recursive=True,
        ),
        output=OutputConfig(
            root_path=str(tmp_dirs["output"]),
            formats=["txt", "json"],
        ),
        ocr=OCRConfig(engine="tesseract", language="eng"),
        preprocessing=PreprocessingConfig(
            enabled=True,
            min_dpi=200,
            target_dpi=300,
            default_dpi=300,
            deskew=True,
            deskew_threshold_degrees=0.5,
            denoise=False,  # disabled for test speed
            binarization="otsu",
        ),
        pipeline=PipelineConfig(
            num_workers=1,
            batch_size=5,
            max_retries=3,
            stale_processing_minutes=60,
            log_dir=str(tmp_dirs["logs"]),
        ),
    )


# ── TIFF images ───────────────────────────────────────────────────────────────


@pytest.fixture()
def normal_dpi_tiff(tmp_dirs):
    """Single-page grayscale TIFF at 300 DPI."""
    img = Image.new("L", (400, 100), color=255)
    path = tmp_dirs["input"] / "normal.tiff"
    img.save(str(path), format="TIFF", dpi=(300, 300))
    return path


@pytest.fixture()
def low_dpi_tiff(tmp_dirs):
    """Single-page grayscale TIFF at 100 DPI — triggers the upscale branch."""
    img = Image.new("L", (400, 100), color=255)
    path = tmp_dirs["input"] / "low_dpi.tiff"
    img.save(str(path), format="TIFF", dpi=(100, 100))
    return path


@pytest.fixture()
def rgb_tiff(tmp_dirs):
    """Single-page RGB TIFF — triggers the grayscale conversion branch."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    path = tmp_dirs["input"] / "rgb.tiff"
    img.save(str(path), format="TIFF", dpi=(300, 300))
    return path


@pytest.fixture()
def multipage_tiff(tmp_dirs):
    """Three-page TIFF for page-iteration tests."""
    pages = [Image.new("L", (400, 100), color=255) for _ in range(3)]
    path = tmp_dirs["input"] / "multipage.tiff"
    pages[0].save(
        str(path),
        format="TIFF",
        save_all=True,
        append_images=pages[1:],
        dpi=(300, 300),
    )
    return path


@pytest.fixture()
def ocr_tiff(tmp_dirs):
    """TIFF with large, high-contrast text suitable for Tesseract."""
    img = Image.new("RGB", (800, 150), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    draw.text((20, 40), "HELLO OCR WORLD", fill="black", font=font)
    path = tmp_dirs["input"] / "ocr_sample.tiff"
    img.save(str(path), format="TIFF", dpi=(300, 300))
    return path


# ── Mock OCR engine ───────────────────────────────────────────────────────────


class MockOCREngine:
    """Deterministic OCR stub — returns configurable text without Tesseract."""

    def __init__(self, text: str = "Hello OCR World", confidence: float = 0.95):
        self._text = text
        self._confidence = confidence

    def process_page(self, image: Image.Image) -> PageOCRResult:
        return PageOCRResult(
            text=self._text,
            confidence=self._confidence,
            engine="mock",
        )


@pytest.fixture()
def mock_ocr_engine():
    return MockOCREngine()


@pytest.fixture()
def empty_text_ocr_engine():
    """Returns empty text — exercises the falsy-text branch in process_image."""
    return MockOCREngine(text="", confidence=0.0)


# ── Mock engine builder (for T-SQL-heavy functions) ───────────────────────────


def make_mock_engine(rowcount: int = 1) -> MagicMock:
    """Return a MagicMock SQLAlchemy engine where execute().rowcount == rowcount."""
    mock_result = MagicMock()
    mock_result.rowcount = rowcount
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine
