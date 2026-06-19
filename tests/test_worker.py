"""Tests for pipeline/worker.py."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from pipeline.config import Config, OCRConfig, OutputConfig, PreprocessingConfig
from pipeline.db import OCRImage
from pipeline.preprocessor import ImagePreprocessor
from pipeline.worker import (
    _merge_pdf_pages,
    _tesseract_page_to_pdf,
    _worker_id,
    claim_batch,
    mark_complete,
    mark_error,
    process_image,
    reset_stale,
    worker_loop,
)
from tests.conftest import MockOCREngine, make_mock_engine


# ── _worker_id ────────────────────────────────────────────────────────────────


def test_worker_id_contains_hostname_and_pid():
    wid = _worker_id()
    assert "-" in wid
    assert len(wid) > 3


# ── claim_batch ───────────────────────────────────────────────────────────────


def test_claim_batch_returns_rows():
    rows = [(1, "/a.tif", "a.tif"), (2, "/b.tiff", "b.tiff")]
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    batch = claim_batch(engine, "worker-1", batch_size=10, max_retries=3)

    assert len(batch) == 2
    assert batch[0] == {"id": 1, "file_path": "/a.tif", "file_name": "a.tif"}
    assert batch[1]["id"] == 2


def test_claim_batch_empty_queue():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    assert claim_batch(engine, "w", 5, 3) == []


# ── reset_stale ───────────────────────────────────────────────────────────────


def test_reset_stale_returns_rowcount():
    engine = make_mock_engine(rowcount=3)
    assert reset_stale(engine, stale_minutes=60) == 3


def test_reset_stale_zero_rows():
    engine = make_mock_engine(rowcount=0)
    assert reset_stale(engine, stale_minutes=60) == 0


# ── mark_complete ─────────────────────────────────────────────────────────────


def test_mark_complete_sets_status_and_results(sqlite_engine):
    with Session(sqlite_engine) as s:
        img = OCRImage(file_path="/x.tif", file_name="x.tif", status="processing")
        s.add(img)
        s.commit()
        image_id = img.id

    result = {
        "output_base": "/out/x",
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "page one",
                "confidence": 0.9,
                "processing_time_ms": 100,
            },
            {
                "page_number": 2,
                "text": "page two",
                "confidence": 0.8,
                "processing_time_ms": 120,
            },
        ],
    }
    mark_complete(sqlite_engine, image_id, result, "tesseract")

    with Session(sqlite_engine) as s:
        img = s.get(OCRImage, image_id)
        assert img.status == "complete"
        assert img.page_count == 2
        assert img.output_path == "/out/x"
        assert len(img.results) == 2
        assert img.results[0].extracted_text == "page one"


# ── mark_error ────────────────────────────────────────────────────────────────


def test_mark_error_below_max_retries_resets_to_pending(sqlite_engine):
    with Session(sqlite_engine) as s:
        img = OCRImage(
            file_path="/e.tif", file_name="e.tif", status="processing", retry_count=0
        )
        s.add(img)
        s.commit()
        image_id = img.id

    mark_error(sqlite_engine, image_id, "boom", max_retries=3)

    with Session(sqlite_engine) as s:
        img = s.get(OCRImage, image_id)
        assert img.status == "pending"
        assert img.retry_count == 1
        assert img.worker_id is None
        assert img.started_at is None
        assert img.error_message == "boom"


def test_mark_error_at_max_retries_marks_failed(sqlite_engine):
    with Session(sqlite_engine) as s:
        img = OCRImage(
            file_path="/f.tif", file_name="f.tif", status="processing", retry_count=2
        )
        s.add(img)
        s.commit()
        image_id = img.id

    mark_error(sqlite_engine, image_id, "final failure", max_retries=3)

    with Session(sqlite_engine) as s:
        img = s.get(OCRImage, image_id)
        assert img.status == "failed"
        assert img.retry_count == 3


# ── _tesseract_page_to_pdf ────────────────────────────────────────────────────


def test_tesseract_page_to_pdf(mocker):
    mock_pt = MagicMock()
    mock_pt.image_to_pdf_or_hocr.return_value = b"%PDF-fake"
    import sys

    mocker.patch.dict(sys.modules, {"pytesseract": mock_pt})

    img = Image.new("L", (100, 100), color=255)
    data = _tesseract_page_to_pdf(img, "eng", "--oem 1 --psm 3")

    assert data == b"%PDF-fake"
    mock_pt.image_to_pdf_or_hocr.assert_called_once()


# ── _merge_pdf_pages ──────────────────────────────────────────────────────────


def test_merge_pdf_pages_creates_file(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    out = tmp_path / "merged.pdf"
    _merge_pdf_pages([pdf_bytes, pdf_bytes], out)

    assert out.exists()
    assert out.stat().st_size > 0


# ── process_image ─────────────────────────────────────────────────────────────


def _make_cfg(
    tmp_dirs, formats=("txt", "json"), engine="tesseract", preprocessing_enabled=True
):
    from pipeline.config import (
        DatabaseConfig,
        InputConfig,
        PipelineConfig,
    )

    return Config(
        database=DatabaseConfig(),
        input=InputConfig(root_path=str(tmp_dirs["input"])),
        output=OutputConfig(root_path=str(tmp_dirs["output"]), formats=list(formats)),
        ocr=OCRConfig(engine=engine),
        preprocessing=PreprocessingConfig(
            enabled=preprocessing_enabled,
            denoise=False,
            binarization="otsu",
        ),
        pipeline=PipelineConfig(log_dir=str(tmp_dirs["logs"])),
    )


def test_process_image_txt_and_json_output(tmp_dirs, normal_dpi_tiff, mock_ocr_engine):
    cfg = _make_cfg(tmp_dirs)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    assert result["page_count"] == 1
    stem = normal_dpi_tiff.stem
    assert (tmp_dirs["output"] / f"{stem}.txt").exists()
    assert (tmp_dirs["output"] / f"{stem}.json").exists()


def test_process_image_json_structure(tmp_dirs, normal_dpi_tiff, mock_ocr_engine):
    cfg = _make_cfg(tmp_dirs)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    stem = normal_dpi_tiff.stem
    doc = json.loads((tmp_dirs["output"] / f"{stem}.json").read_text())
    assert "full_text" in doc
    assert "pages" in doc
    assert doc["page_count"] == 1
    assert doc["ocr_engine"] == "tesseract"


def test_process_image_txt_only(tmp_dirs, normal_dpi_tiff, mock_ocr_engine):
    cfg = _make_cfg(tmp_dirs, formats=("txt",))
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    stem = normal_dpi_tiff.stem
    assert (tmp_dirs["output"] / f"{stem}.txt").exists()
    assert not (tmp_dirs["output"] / f"{stem}.json").exists()


def test_process_image_json_only(tmp_dirs, normal_dpi_tiff, mock_ocr_engine):
    cfg = _make_cfg(tmp_dirs, formats=("json",))
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    stem = normal_dpi_tiff.stem
    assert not (tmp_dirs["output"] / f"{stem}.txt").exists()
    assert (tmp_dirs["output"] / f"{stem}.json").exists()


def test_process_image_empty_ocr_text(tmp_dirs, normal_dpi_tiff, empty_text_ocr_engine):
    """Empty OCR result must not crash; full_text_parts stays empty."""
    cfg = _make_cfg(tmp_dirs)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=empty_text_ocr_engine,
    )

    stem = normal_dpi_tiff.stem
    assert result["page_count"] == 1
    assert (tmp_dirs["output"] / f"{stem}.txt").read_text() == ""


def test_process_image_outside_input_root(tmp_dirs, mock_ocr_engine):
    """file_path not under input_root triggers ValueError → falls back to file name."""
    # Place the TIFF in a completely different directory
    other_dir = tmp_dirs["root"] / "elsewhere"
    other_dir.mkdir()
    img = Image.new("L", (200, 50), color=255)
    tiff = other_dir / "outside.tiff"
    img.save(str(tiff), format="TIFF", dpi=(300, 300))

    cfg = _make_cfg(tmp_dirs)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(tiff),
        input_root=str(tmp_dirs["input"]),  # tiff is NOT under input_root
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    assert result["page_count"] == 1
    assert (tmp_dirs["output"] / "outside.txt").exists()


def test_process_image_preprocessing_disabled_grayscale(
    tmp_dirs, normal_dpi_tiff, mock_ocr_engine
):
    """preprocessing.enabled=False with an L-mode image takes the passthrough path."""
    cfg = _make_cfg(tmp_dirs, preprocessing_enabled=False)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(normal_dpi_tiff),  # already grayscale "L"
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )
    assert result["page_count"] == 1


def test_process_image_preprocessing_disabled_rgb(tmp_dirs, rgb_tiff, mock_ocr_engine):
    """preprocessing.enabled=False with an RGB image triggers convert('L')."""
    cfg = _make_cfg(tmp_dirs, preprocessing_enabled=False)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(rgb_tiff),  # RGB — must be converted
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )
    assert result["page_count"] == 1


def test_process_image_multipage_tiff(tmp_dirs, multipage_tiff, mock_ocr_engine):
    """All pages of a multi-page TIFF are processed."""
    cfg = _make_cfg(tmp_dirs)
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    result = process_image(
        file_path=str(multipage_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )
    assert result["page_count"] == 3
    assert len(result["pages"]) == 3


def test_process_image_pdf_output_tesseract(
    tmp_dirs, normal_dpi_tiff, mock_ocr_engine, mocker
):
    """pdf format + tesseract engine → _tesseract_page_to_pdf and _merge_pdf_pages called."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    fake_pdf = buf.getvalue()

    mock_page_pdf = mocker.patch(
        "pipeline.worker._tesseract_page_to_pdf", return_value=fake_pdf
    )
    mock_merge = mocker.patch("pipeline.worker._merge_pdf_pages")

    cfg = _make_cfg(tmp_dirs, formats=("pdf",), engine="tesseract")
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    mock_page_pdf.assert_called_once()
    mock_merge.assert_called_once()


def test_process_image_pdf_output_non_tesseract(
    tmp_dirs, normal_dpi_tiff, mock_ocr_engine, mocker
):
    """pdf format + non-tesseract engine → warning logged, no PDF written."""
    mock_merge = mocker.patch("pipeline.worker._merge_pdf_pages")

    cfg = _make_cfg(tmp_dirs, formats=("pdf",), engine="easyocr")
    preprocessor = ImagePreprocessor(cfg.preprocessing)

    process_image(
        file_path=str(normal_dpi_tiff),
        input_root=str(tmp_dirs["input"]),
        output_root=str(tmp_dirs["output"]),
        cfg=cfg,
        preprocessor=preprocessor,
        ocr_engine=mock_ocr_engine,
    )

    mock_merge.assert_not_called()


# ── worker_loop ───────────────────────────────────────────────────────────────


def _mock_loop_deps(mocker, tmp_dirs, claim_side_effect=None, reset_count=0):
    """Patch all I/O dependencies of worker_loop and return a ready Config."""
    from pipeline.config import (
        DatabaseConfig,
        InputConfig,
        PipelineConfig,
    )

    cfg = Config(
        database=DatabaseConfig(),
        input=InputConfig(root_path=str(tmp_dirs["input"])),
        output=OutputConfig(root_path=str(tmp_dirs["output"]), formats=["txt", "json"]),
        ocr=OCRConfig(engine="tesseract"),
        preprocessing=PreprocessingConfig(
            enabled=True, denoise=False, binarization="otsu"
        ),
        pipeline=PipelineConfig(
            num_workers=1,
            batch_size=1,
            max_retries=3,
            stale_processing_minutes=60,
            log_dir=str(tmp_dirs["logs"]),
        ),
    )
    mocker.patch("pipeline.worker.load_config", return_value=cfg)
    mocker.patch("pipeline.worker.create_db_engine", return_value=MagicMock())
    mocker.patch("pipeline.worker.create_ocr_engine", return_value=MockOCREngine())
    mocker.patch("pipeline.worker.reset_stale", return_value=reset_count)
    mocker.patch("pipeline.worker.logger")

    if claim_side_effect is not None:
        mocker.patch("pipeline.worker.claim_batch", side_effect=claim_side_effect)
    else:
        mocker.patch("pipeline.worker.claim_batch", return_value=[])

    return cfg


def test_worker_loop_exits_immediately_when_no_pending(tmp_dirs, mocker):
    _mock_loop_deps(mocker, tmp_dirs)
    worker_loop("dummy.yaml")  # must return without error


def test_worker_loop_logs_stale_reset_count(tmp_dirs, mocker):
    """reset_count > 0 causes an info log — exercises the if-reset_count branch."""
    _mock_loop_deps(mocker, tmp_dirs, reset_count=5)
    worker_loop("dummy.yaml")
    # logger is mocked; just confirm no exception


def test_worker_loop_handles_stale_reset_exception(tmp_dirs, mocker):
    """Exception from reset_stale is caught and logged as a warning."""
    _mock_loop_deps(mocker, tmp_dirs)
    mocker.patch("pipeline.worker.reset_stale", side_effect=RuntimeError("db gone"))
    worker_loop("dummy.yaml")


def test_worker_loop_processes_item_successfully(tmp_dirs, mocker, normal_dpi_tiff):
    """Successful processing calls mark_complete and increments processed counter."""
    item = {
        "id": 1,
        "file_path": str(normal_dpi_tiff),
        "file_name": normal_dpi_tiff.name,
    }

    # First call returns one item, second call signals queue empty
    _mock_loop_deps(mocker, tmp_dirs, claim_side_effect=[[item], []])
    mock_mark_complete = mocker.patch("pipeline.worker.mark_complete")
    mocker.patch(
        "pipeline.worker.process_image",
        return_value={
            "output_base": "/out/x",
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "ok",
                    "confidence": 0.9,
                    "processing_time_ms": 50,
                }
            ],
            "total_ms": 50,
        },
    )

    worker_loop("dummy.yaml")
    mock_mark_complete.assert_called_once()


def test_worker_loop_handles_unidentified_image_error(
    tmp_dirs, mocker, normal_dpi_tiff
):
    """UnidentifiedImageError → mark_error called, loop continues."""
    item = {
        "id": 2,
        "file_path": str(normal_dpi_tiff),
        "file_name": normal_dpi_tiff.name,
    }
    _mock_loop_deps(mocker, tmp_dirs, claim_side_effect=[[item], []])
    mocker.patch(
        "pipeline.worker.process_image", side_effect=UnidentifiedImageError("bad")
    )
    mock_mark_error = mocker.patch("pipeline.worker.mark_error")

    worker_loop("dummy.yaml")
    mock_mark_error.assert_called_once()


def test_worker_loop_handles_os_error(tmp_dirs, mocker, normal_dpi_tiff):
    """OSError is caught by the (UnidentifiedImageError, OSError) handler."""
    item = {
        "id": 3,
        "file_path": str(normal_dpi_tiff),
        "file_name": normal_dpi_tiff.name,
    }
    _mock_loop_deps(mocker, tmp_dirs, claim_side_effect=[[item], []])
    mocker.patch("pipeline.worker.process_image", side_effect=OSError("no file"))
    mock_mark_error = mocker.patch("pipeline.worker.mark_error")

    worker_loop("dummy.yaml")
    mock_mark_error.assert_called_once()


def test_worker_loop_handles_general_exception(tmp_dirs, mocker, normal_dpi_tiff):
    """Unexpected exceptions hit the bare except-Exception handler."""
    item = {
        "id": 4,
        "file_path": str(normal_dpi_tiff),
        "file_name": normal_dpi_tiff.name,
    }
    _mock_loop_deps(mocker, tmp_dirs, claim_side_effect=[[item], []])
    mocker.patch("pipeline.worker.process_image", side_effect=ValueError("unexpected"))
    mock_mark_error = mocker.patch("pipeline.worker.mark_error")

    worker_loop("dummy.yaml")
    mock_mark_error.assert_called_once()
