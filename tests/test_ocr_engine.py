"""
Tests for pipeline/ocr_engine.py.

EasyOCR and PaddleOCR are not installed in the test environment, so their
modules are injected into sys.modules as MagicMocks.  This lets us test the
wrapper classes without the heavy GPU libraries being present.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pipeline.config import OCRConfig
from pipeline.ocr_engine import (
    EasyOCREngine,
    PaddleOCREngine,
    PageOCRResult,
    TesseractEngine,
    create_engine,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _blank_image() -> Image.Image:
    return Image.new("L", (100, 100), color=200)


def _mock_pytesseract(mocker, texts=("Hello World",), confs=(85, 90, -1)):
    """Patch pytesseract in the ocr_engine module namespace."""
    mock_pt = MagicMock()
    mock_pt.Output.DICT = "dict"
    mock_pt.image_to_data.return_value = {"conf": list(confs)}
    mock_pt.image_to_string.return_value = " ".join(texts) + "\n"
    mocker.patch.dict(sys.modules, {"pytesseract": mock_pt})
    mocker.patch("pipeline.ocr_engine.TesseractEngine.__init__.__globals__", {})
    return mock_pt


# ── create_engine factory ─────────────────────────────────────────────────────


def test_create_engine_returns_tesseract(mocker):
    mocker.patch.dict(sys.modules, {"pytesseract": MagicMock()})
    cfg = OCRConfig(engine="tesseract", tesseract_cmd=None)
    engine = create_engine(cfg)
    assert isinstance(engine, TesseractEngine)


def test_create_engine_sets_tesseract_cmd(mocker):
    mock_pt = MagicMock()
    mocker.patch.dict(sys.modules, {"pytesseract": mock_pt})
    cfg = OCRConfig(engine="tesseract", tesseract_cmd="/usr/bin/tesseract")
    create_engine(cfg)
    # tesseract_cmd attribute on pytesseract.pytesseract must be set
    assert mock_pt.pytesseract.tesseract_cmd == "/usr/bin/tesseract"


def test_create_engine_returns_easyocr(mocker):
    mock_easyocr = MagicMock()
    mocker.patch.dict(sys.modules, {"easyocr": mock_easyocr, "numpy": MagicMock()})
    cfg = OCRConfig(engine="easyocr", language="eng")
    engine = create_engine(cfg)
    assert isinstance(engine, EasyOCREngine)


def test_create_engine_returns_paddleocr(mocker):
    mock_paddle = MagicMock()
    mocker.patch.dict(sys.modules, {"paddleocr": mock_paddle, "numpy": MagicMock()})
    cfg = OCRConfig(engine="paddleocr", language="eng")
    engine = create_engine(cfg)
    assert isinstance(engine, PaddleOCREngine)


def test_create_engine_unknown_raises():
    cfg = OCRConfig(engine="nonexistent_engine")
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        create_engine(cfg)


# ── TesseractEngine ───────────────────────────────────────────────────────────


def _make_tesseract_engine(mocker, confs=(80, 90, -1), text="Hello World"):
    """Build a TesseractEngine with a fully mocked pytesseract."""
    mock_pt = MagicMock()
    mock_pt.Output.DICT = "dict"
    mock_pt.image_to_data.return_value = {"conf": list(confs)}
    mock_pt.image_to_string.return_value = text + "\n"
    mocker.patch.dict(sys.modules, {"pytesseract": mock_pt})

    cfg = OCRConfig(engine="tesseract")
    engine = TesseractEngine.__new__(TesseractEngine)
    engine._pt = mock_pt
    engine._lang = cfg.language
    engine._config = cfg.tesseract_config
    return engine


def test_tesseract_process_page_returns_result(mocker):
    engine = _make_tesseract_engine(mocker, confs=(80, 90, -1), text="Hello World")
    result = engine.process_page(_blank_image())

    assert isinstance(result, PageOCRResult)
    assert result.text == "Hello World"
    assert result.engine == "tesseract"
    assert 0.0 <= result.confidence <= 1.0


def test_tesseract_process_page_all_invalid_confs(mocker):
    """All conf == -1 → mean_conf defaults to 0.0."""
    engine = _make_tesseract_engine(mocker, confs=(-1, -1, -1), text="sparse")
    result = engine.process_page(_blank_image())
    assert result.confidence == pytest.approx(0.0)


def test_tesseract_process_page_strips_whitespace(mocker):
    engine = _make_tesseract_engine(mocker, confs=(70,), text="  trimmed  ")
    result = engine.process_page(_blank_image())
    assert result.text == "trimmed"


# ── EasyOCREngine ─────────────────────────────────────────────────────────────


def _make_easyocr_engine(mocker, read_results=None):
    mock_easyocr = MagicMock()
    mocker.patch.dict(sys.modules, {"easyocr": mock_easyocr})

    # Lightweight numpy-like stub used for tests (avoids importing C extensions)
    class _NpStub:
        def array(self, img):
            return img

    np = _NpStub()

    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine._np = np
    engine._reader = MagicMock()
    if read_results is not None:
        engine._reader.readtext.return_value = read_results
    else:
        engine._reader.readtext.return_value = []
    return engine


def test_easyocr_process_page_with_results(mocker):
    read_results = [
        (None, "hello", 0.9),
        (None, "world", 0.8),
    ]
    engine = _make_easyocr_engine(mocker, read_results=read_results)
    result = engine.process_page(_blank_image())

    assert result.text == "hello world"
    assert result.confidence == pytest.approx(0.85)
    assert result.engine == "easyocr"


def test_easyocr_process_page_empty_results(mocker):
    """No detected text → empty string and 0.0 confidence."""
    engine = _make_easyocr_engine(mocker, read_results=[])
    result = engine.process_page(_blank_image())

    assert result.text == ""
    assert result.confidence == pytest.approx(0.0)


# ── PaddleOCREngine ───────────────────────────────────────────────────────────


def _make_paddle_engine(mocker, ocr_results=None):
    mock_paddle = MagicMock()
    mocker.patch.dict(sys.modules, {"paddleocr": mock_paddle})

    # Lightweight numpy-like stub used for tests (avoids importing C extensions)
    class _NpStub:
        def array(self, img):
            return img

    np = _NpStub()

    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine._np = np
    engine._ocr = MagicMock()
    engine._ocr.ocr.return_value = ocr_results
    return engine


def test_paddleocr_process_page_with_results(mocker):
    # PaddleOCR returns: [[bounding_box, (text, confidence)], ...]
    ocr_results = [
        [
            ([0, 0, 10, 10], ("line one", 0.95)),
            ([0, 20, 10, 30], ("line two", 0.85)),
        ]
    ]
    engine = _make_paddle_engine(mocker, ocr_results=ocr_results)
    result = engine.process_page(_blank_image())

    assert "line one" in result.text
    assert result.confidence == pytest.approx(0.9)
    assert result.engine == "paddleocr"


def test_paddleocr_process_page_none_results(mocker):
    """ocr() returns None (empty document)."""
    engine = _make_paddle_engine(mocker, ocr_results=None)
    result = engine.process_page(_blank_image())
    assert result.text == ""
    assert result.confidence == pytest.approx(0.0)


def test_paddleocr_process_page_empty_inner_list(mocker):
    """ocr() returns [[]] (page with no lines)."""
    engine = _make_paddle_engine(mocker, ocr_results=[[]])
    result = engine.process_page(_blank_image())
    assert result.text == ""
    assert result.confidence == pytest.approx(0.0)
