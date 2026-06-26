from __future__ import annotations

from unittest.mock import MagicMock

import os
from PIL import Image

from pipeline.preprocessor import ImagePreprocessor
from pipeline.config import PreprocessingConfig
from pipeline.worker import claim_batch, reset_stale, _tesseract_page_to_pdf


def _preprocessor(**kwargs) -> ImagePreprocessor:
    return ImagePreprocessor(PreprocessingConfig(**kwargs))


def test_upscale_noop_at_same_dpi():
    img = Image.new("L", (100, 50))
    pp = _preprocessor(target_dpi=300)
    out = pp._upscale(img, current_dpi=300)
    # When scale == 1.0 we return the original object
    assert out is img


def test_upscale_skips_when_scale_too_large():
    img = Image.new("L", (100, 50))
    pp = _preprocessor(target_dpi=10000)
    out = pp._upscale(img, current_dpi=100)
    # scale == 100 -> >5.0 -> skip upscaling
    assert out is img


def test_upscale_skips_when_resulting_size_too_large():
    class DummyPage:
        def __init__(self, w, h):
            self.width = w
            self.height = h

        def resize(self, size, resample):
            return "resized"

    # Set a small current_dpi so that computed new_w*new_h becomes huge
    dummy = DummyPage(100000, 100000)
    pp = _preprocessor(target_dpi=300)
    out = pp._upscale(dummy, current_dpi=1)
    assert out is dummy


def test_claim_batch_sqlserver_branch():
    rows = [(1, "/a.tif", "a.tif")]
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_conn.engine = MagicMock()
    mock_conn.engine.dialect = MagicMock()
    mock_conn.engine.dialect.name = "mssql"
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    batch = claim_batch(engine, "w", batch_size=5, max_retries=3)
    assert isinstance(batch, list)
    assert batch[0]["id"] == rows[0]


def test_reset_stale_sqlserver_branch():
    mock_result = MagicMock()
    mock_result.rowcount = 7
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    mock_conn.engine = MagicMock()
    mock_conn.engine.dialect = MagicMock()
    mock_conn.engine.dialect.name = "mssql"
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    assert reset_stale(engine, stale_minutes=15) == 7


def test_tesseract_page_to_pdf_accepts_config_object(mocker, tmp_path):
    mock_pt = MagicMock()
    mock_pt.image_to_pdf_or_hocr.return_value = b"%PDF-fake"
    mocker.patch.dict("sys.modules", {"pytesseract": mock_pt})

    class Cfg:
        tesseract_config = "--oem 1 --psm 3"
        tessdata_prefix = str(tmp_path)

    img = Image.new("L", (10, 10), color=255)
    data = _tesseract_page_to_pdf(img, "eng", Cfg())
    assert data == b"%PDF-fake"
    # tessdata prefix should be set in environment when provided
    assert os.environ.get("TESSDATA_PREFIX") == str(tmp_path)


def test_process_page_handles_zero_current_dpi(monkeypatch):
    img = Image.new("L", (100, 100))
    pp = _preprocessor()
    # Force _get_dpi to return a bare 0 (simulates broken metadata)
    monkeypatch.setattr(
        "pipeline.preprocessor.ImagePreprocessor._get_dpi", lambda self, i: 0
    )
    res = pp.process_page(img)
    # Should have fallen back to default DPI and returned a ProcessedPage
    assert res is not None


def test_process_page_skips_massive_upscale(monkeypatch):
    class FakeImage:
        def __init__(self):
            self.width = 10000
            self.height = 10000
            self.mode = "L"
            self.info = {}

        def convert(self, mode):
            return self

        def resize(self, size, resample):
            return self

        def __array__(self, dtype=None):
            import numpy as np

            return np.zeros((10, 10), dtype=np.uint8)

    img = FakeImage()
    pp = _preprocessor(min_dpi=200, target_dpi=300)
    # make DPI low so scale != 1 and current_dpi < min_dpi
    monkeypatch.setattr(
        "pipeline.preprocessor.ImagePreprocessor._get_dpi", lambda self, i: (100, 100)
    )
    res = pp.process_page(img)
    assert res.was_upscaled is False


def test_process_page_detects_scale_too_high(monkeypatch):
    img = Image.new("L", (100, 100))
    pp = _preprocessor(min_dpi=200, target_dpi=2000)
    monkeypatch.setattr(
        "pipeline.preprocessor.ImagePreprocessor._get_dpi", lambda self, i: (100, 100)
    )
    res = pp.process_page(img)
    # scale > 5.0 path should be taken and upscaling skipped
    assert res.was_upscaled is False


def test_claim_batch_fallback_extraction_exception():
    class BadRow:
        pass

    rows = [BadRow()]
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    batch = claim_batch(engine, "worker-1", batch_size=10, max_retries=3)
    assert batch[0]["id"] is rows[0]
    assert batch[0]["file_path"] is rows[0]
    assert batch[0]["file_name"] is rows[0]


def test_get_dpi_single_tuple_element():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = (150,)
    pp = _preprocessor()
    assert pp._get_dpi(img) == (150.0, 150.0)


def test_get_dpi_scalar_value():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = 200
    pp = _preprocessor()
    assert pp._get_dpi(img) == (200.0, 200.0)


def test_reset_stale_sqlite_branch():
    mock_result = MagicMock()
    mock_result.rowcount = 5
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    # Provide an engine.dialect.name == 'sqlite' on the connection
    mock_conn.engine = MagicMock()
    mock_conn.engine.dialect = MagicMock()
    mock_conn.engine.dialect.name = "sqlite"
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    assert reset_stale(engine, stale_minutes=30) == 5


def test_get_dpi_single_element_zero():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = (0,)
    pp = _preprocessor(default_dpi=77)
    assert pp._get_dpi(img) == (77.0, 77.0)


def test_get_dpi_scalar_zero():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = 0
    pp = _preprocessor(default_dpi=88)
    assert pp._get_dpi(img) == (88.0, 88.0)


def test_upscale_resulting_size_too_large_trigger():
    class DummyPage:
        def __init__(self, w, h):
            self.width = w
            self.height = h

        def resize(self, size, resample):
            return "resized"

    # Choose current_dpi so scale <= 5.0 but resulting size > 20_000_000
    dummy = DummyPage(10_000, 10_000)
    pp = _preprocessor(target_dpi=300)
    out = pp._upscale(dummy, current_dpi=100)
    assert out is dummy


def test_get_dpi_negative_scalar():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = -1
    pp = _preprocessor(default_dpi=55)
    assert pp._get_dpi(img) == (55.0, 55.0)


def test_insert_batch_sqlite_branch():
    from pipeline.discovery import _insert_batch

    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    mock_conn.engine = MagicMock()
    mock_conn.engine.dialect = MagicMock()
    mock_conn.engine.dialect.name = "sqlite"
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    batch = [{"file_path": "/a.tif", "file_name": "a.tif", "file_size_bytes": 10}]
    r, s = _insert_batch(engine, batch)
    assert r == 1 and s == 0


def test_build_connection_url_and_create_engine_sqlite(tmp_path):
    from pipeline.db import build_connection_url, create_db_engine
    from pipeline.config import DatabaseConfig

    # Relative path
    cfg = DatabaseConfig(server="", database="mydb.sqlite", driver="sqlite")
    url = build_connection_url(cfg)
    assert url.startswith("sqlite:///")

    # Absolute path
    abs_path = str(tmp_path / "abs.db")
    cfg2 = DatabaseConfig(server="", database=abs_path, driver="sqlite")
    url2 = build_connection_url(cfg2)
    # absolute should still start with sqlite:/// but include the leading slash
    assert url2.startswith("sqlite:///") and abs_path in url2

    # create_db_engine should return an engine for sqlite
    engine = create_db_engine(cfg)
    assert engine is not None
    # engine.dialect.name should be 'sqlite'
    assert getattr(engine.dialect, "name", "") == "sqlite"


def test_get_dpi_invalid_value():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = "not-a-number"
    pp = _preprocessor(default_dpi=66)
    assert pp._get_dpi(img) == (66.0, 66.0)


def test_maybe_upscale_handles_exception(monkeypatch):
    class BadNumber:
        def __bool__(self):
            raise RuntimeError("boom")

    class FakePage:
        def __init__(self):
            self.width = 100
            self.height = 100

        def resize(self, size, resample):
            return self

    pp = _preprocessor()
    fake = FakePage()
    # Passing a BadNumber should trigger the except branch inside _maybe_upscale
    page_out, was_up = pp._maybe_upscale(fake, BadNumber())
    assert page_out is fake
    assert was_up is False


def test_get_dpi_two_element_tuple():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = (150, 160)
    pp = _preprocessor()
    assert pp._get_dpi(img) == (150.0, 160.0)


def test_get_dpi_exhaustive_cases():
    pp = _preprocessor(default_dpi=42)
    cases = [
        (150, 160),
        (0, 160),
        (150, 0),
        (150,),
        (0,),
        200,
        0,
        -5,
        "invalid",
        None,
    ]
    for val in cases:
        img = Image.new("L", (10, 10))
        if val is not None:
            img.info["dpi"] = val
        out = pp._get_dpi(img)
        assert isinstance(out, tuple) and len(out) == 2


def test_get_dpi_two_element_invalid():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = ("not-a-number", "also-bad")
    pp = _preprocessor(default_dpi=99)
    assert pp._get_dpi(img) == (99.0, 99.0)


def test_get_dpi_single_element_invalid():
    img = Image.new("L", (100, 100))

    class Bad:
        def __str__(self):
            raise RuntimeError("boom")

    img.info["dpi"] = (Bad(),)
    pp = _preprocessor(default_dpi=101)
    assert pp._get_dpi(img) == (101.0, 101.0)


def test_get_dpi_empty_tuple():
    img = Image.new("L", (100, 100))
    img.info["dpi"] = ()
    pp = _preprocessor(default_dpi=72)
    assert pp._get_dpi(img) == (72.0, 72.0)
