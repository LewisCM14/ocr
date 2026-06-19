"""
Worker
~~~~~~
Each worker process runs worker_loop(), which:
  1. Claims a batch of pending images from the database atomically.
  2. For each image: preprocesses → OCR → writes output files → updates DB.
  3. Repeats until no pending images remain.

The worker is self-contained: it creates its own DB connection and OCR engine
so it can safely run in a separate OS process (no shared state).
"""

from __future__ import annotations

import io
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .config import Config, load_config
from .db import OCRImage, OCRResult, create_db_engine
from .ocr_engine import create_engine as create_ocr_engine
from .preprocessor import ImagePreprocessor


# ── Worker identity ───────────────────────────────────────────────────────────


def _worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


# ── Database helpers ──────────────────────────────────────────────────────────


def claim_batch(
    engine: Engine,
    worker_id: str,
    batch_size: int,
    max_retries: int,
) -> list[dict]:
    """
    Atomically claim up to `batch_size` pending images by updating their status
    to 'processing' and returning their ids/paths.

    The OUTPUT clause ensures claiming is race-condition-free even with many
    concurrent workers hitting the same MSSQL instance.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                UPDATE TOP ({batch_size}) ocr_images
                SET
                    status     = 'processing',
                    worker_id  = :worker_id,
                    started_at = GETUTCDATE()
                OUTPUT
                    INSERTED.id,
                    INSERTED.file_path,
                    INSERTED.file_name
                WHERE
                    status      = 'pending'
                    AND retry_count < :max_retries
                """
            ),
            {"worker_id": worker_id, "max_retries": max_retries},
        ).fetchall()
    return [{"id": r[0], "file_path": r[1], "file_name": r[2]} for r in rows]


def reset_stale(engine: Engine, stale_minutes: int) -> int:
    """
    Reset images that have been stuck in 'processing' longer than `stale_minutes`.
    This recovers work items from workers that crashed without updating the DB.
    Returns the number of rows reset.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE ocr_images
                SET
                    status     = 'pending',
                    worker_id  = NULL,
                    started_at = NULL,
                    retry_count = retry_count + 1
                WHERE
                    status      = 'processing'
                    AND started_at < DATEADD(MINUTE, :neg_minutes, GETUTCDATE())
                """
            ),
            {"neg_minutes": -stale_minutes},
        )
    return result.rowcount


def mark_complete(
    engine: Engine, image_id: int, result: dict, ocr_engine_name: str
) -> None:
    with Session(engine) as session:
        image = session.get(OCRImage, image_id)
        image.status = "complete"
        image.completed_at = datetime.now(timezone.utc)
        image.output_path = result["output_base"]
        image.page_count = result["page_count"]

        for page in result["pages"]:
            session.add(
                OCRResult(
                    image_id=image_id,
                    page_number=page["page_number"],
                    extracted_text=page["text"],
                    confidence_score=page["confidence"],
                    processing_time_ms=page["processing_time_ms"],
                    ocr_engine=ocr_engine_name,
                )
            )
        session.commit()


def mark_error(engine: Engine, image_id: int, error: str, max_retries: int) -> None:
    with Session(engine) as session:
        image = session.get(OCRImage, image_id)
        image.retry_count = (image.retry_count or 0) + 1
        if image.retry_count >= max_retries:
            image.status = "failed"
        else:
            image.status = "pending"
            image.worker_id = None
            image.started_at = None
        image.error_message = error[:4000]  # guard against oversized tracebacks
        session.commit()


# ── Single-image processing ───────────────────────────────────────────────────


def _compute_paths(file_path: str, input_root: str, output_root: str):
    src = Path(file_path)
    try:
        rel = src.relative_to(input_root)
    except ValueError:
        rel = Path(src.name)

    out_dir = Path(output_root) / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    return src, rel, out_dir, stem


def _determine_write_pdf(cfg: Config) -> bool:
    write_pdf = "pdf" in cfg.output.formats
    if write_pdf and cfg.ocr.engine != "tesseract":
        logger.warning(
            "Searchable PDF output requires ocr.engine = 'tesseract'. "
            "PDF will not be written for this image."
        )
        write_pdf = False
    return write_pdf


def _get_ocr_input(
    page_frame: Image.Image, cfg: Config, preprocessor: ImagePreprocessor
) -> Image.Image:
    if cfg.preprocessing.enabled:
        processed = preprocessor.process_page(page_frame)
        return processed.image
    return page_frame.convert("L") if page_frame.mode != "L" else page_frame


def _process_pages(
    img: Image.Image,
    cfg: Config,
    preprocessor: ImagePreprocessor,
    ocr_engine: Any,
    write_pdf: bool,
):
    pages: list[dict] = []
    full_text_parts: list[str] = []
    pdf_pages: list[bytes] = []
    page_num = 0

    try:
        while True:
            page_num += 1
            page_frame = img.copy()
            ocr_input = _get_ocr_input(page_frame, cfg, preprocessor)

            page_t0 = time.monotonic()
            ocr_result = ocr_engine.process_page(ocr_input)
            page_ms = int((time.monotonic() - page_t0) * 1000)

            pages.append(
                {
                    "page_number": page_num,
                    "text": ocr_result.text,
                    "confidence": ocr_result.confidence,
                    "word_count": len(ocr_result.text.split())
                    if ocr_result.text
                    else 0,
                    "char_count": len(ocr_result.text) if ocr_result.text else 0,
                    "processing_time_ms": page_ms,
                }
            )

            if ocr_result.text:
                full_text_parts.append(ocr_result.text)

            if write_pdf:
                pdf_pages.append(
                    _tesseract_page_to_pdf(
                        ocr_input, cfg.ocr.language, cfg.ocr.tesseract_config
                    )
                )

            img.seek(img.tell() + 1)
    except EOFError:
        pass

    return pages, full_text_parts, pdf_pages, page_num


def process_image(
    file_path: str,
    input_root: str,
    output_root: str,
    cfg: Config,
    preprocessor: ImagePreprocessor,
    ocr_engine: Any,
) -> dict:
    """Process all pages of one TIFF file and return registration info."""
    wall_start = time.monotonic()

    src, rel, out_dir, stem = _compute_paths(file_path, input_root, output_root)
    write_pdf = _determine_write_pdf(cfg)

    img = Image.open(file_path)
    pages, full_text_parts, pdf_pages, page_num = _process_pages(
        img, cfg, preprocessor, ocr_engine, write_pdf
    )

    total_ms = int((time.monotonic() - wall_start) * 1000)
    full_text = "\n\n--- Page Break ---\n\n".join(full_text_parts)
    output_base = str(out_dir / stem)

    if write_pdf and pdf_pages:
        _merge_pdf_pages(pdf_pages, out_dir / f"{stem}.pdf")

    if "txt" in cfg.output.formats:
        txt_path = out_dir / f"{stem}.txt"
        txt_path.write_text(full_text, encoding="utf-8")

    if "json" in cfg.output.formats:
        json_doc = {
            "source_path": file_path,
            "file_name": src.name,
            "relative_path": str(rel).replace("\\", "/"),
            "file_size_bytes": src.stat().st_size,
            "page_count": page_num,
            "full_text": full_text,
            "pages": pages,
            "ocr_engine": cfg.ocr.engine,
            "ocr_language": cfg.ocr.language,
            "ocr_config": cfg.ocr.tesseract_config,
            "preprocessing_enabled": cfg.preprocessing.enabled,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "total_processing_time_ms": total_ms,
        }
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(json_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {
        "output_base": output_base,
        "page_count": page_num,
        "pages": pages,
        "total_ms": total_ms,
    }


# ── Searchable PDF helpers ───────────────────────────────────────────────────


def _tesseract_page_to_pdf(image: Any, language: str, config: str) -> bytes:
    """
    Ask Tesseract to render a single PIL Image as a PDF with an embedded text
    layer and return the raw PDF bytes.  The text layer is invisible to the
    reader but fully searchable; the visual content is the preprocessed image.
    """
    import pytesseract

    return pytesseract.image_to_pdf_or_hocr(
        image,
        lang=language,
        config=config,
        extension="pdf",
    )


def _merge_pdf_pages(pdf_pages: list[bytes], output_path: Path) -> None:
    """
    Merge a list of single-page PDF byte strings into one multi-page PDF file
    using pypdf (pure Python, no external binary required).
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    for page_bytes in pdf_pages:
        reader_buf = io.BytesIO(page_bytes)
        writer.append(reader_buf)

    with open(output_path, "wb") as fh:
        writer.write(fh)


# ── Worker loop (entry point for each subprocess) ─────────────────────────────


def worker_loop(config_path: str) -> None:
    """
    Main loop executed by each worker process.
    Runs until there are no more pending images.
    """
    cfg = load_config(config_path)
    worker_id = _worker_id()

    # Per-worker log file
    log_dir = Path(cfg.pipeline.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / f"worker_{worker_id}.log",
        rotation="100 MB",
        retention="30 days",
        level=cfg.pipeline.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.info(f"Worker {worker_id} starting")

    engine = create_db_engine(cfg.database)
    preprocessor = ImagePreprocessor(cfg.preprocessing)
    ocr_eng = create_ocr_engine(cfg.ocr)

    # One-time: reset items stranded in 'processing' from previous crashes
    try:
        reset_count = reset_stale(engine, cfg.pipeline.stale_processing_minutes)
        if reset_count:
            logger.info(
                f"Reset {reset_count} stale 'processing' items back to 'pending'"
            )
    except Exception as exc:
        logger.warning(f"Stale reset failed (non-fatal): {exc}")

    processed = 0
    errors = 0

    while True:
        batch = claim_batch(
            engine, worker_id, cfg.pipeline.batch_size, cfg.pipeline.max_retries
        )
        if not batch:
            logger.info(f"Worker {worker_id}: no pending images remain — exiting")
            break

        for item in batch:
            image_id: int = item["id"]
            file_path: str = item["file_path"]
            try:
                result = process_image(
                    file_path=file_path,
                    input_root=cfg.input.root_path,
                    output_root=cfg.output.root_path,
                    cfg=cfg,
                    preprocessor=preprocessor,
                    ocr_engine=ocr_eng,
                )
                mark_complete(engine, image_id, result, cfg.ocr.engine)
                processed += 1
                logger.info(
                    f"[OK] {file_path} | "
                    f"pages={result['page_count']} | "
                    f"time={result['total_ms']}ms"
                )
            except OSError as exc:
                errors += 1
                logger.error(f"[FAIL] {file_path} | {exc}")
                mark_error(engine, image_id, str(exc), cfg.pipeline.max_retries)
            except Exception as exc:
                errors += 1
                logger.exception(f"[FAIL] {file_path} | unexpected error: {exc}")
                mark_error(engine, image_id, str(exc), cfg.pipeline.max_retries)

    logger.info(f"Worker {worker_id} finished — processed={processed}, errors={errors}")
