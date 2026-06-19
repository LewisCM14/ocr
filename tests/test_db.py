"""Tests for pipeline/db.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from pipeline.config import DatabaseConfig
from pipeline.db import (
    OCRImage,
    OCRResult,
    build_connection_url,
    create_db_engine,
    init_db,
)


# ── build_connection_url ──────────────────────────────────────────────────────


def test_build_connection_url_trusted():
    cfg = DatabaseConfig(
        server="srv",
        database="db",
        driver="ODBC Driver 17 for SQL Server",
        trusted_connection=True,
    )
    url = build_connection_url(cfg)
    assert url.drivername == "mssql+pyodbc"
    assert url.host == "srv"
    assert url.database == "db"
    assert url.username is None
    assert url.password is None
    assert url.query["trusted_connection"] == "yes"


def test_build_connection_url_sql_auth():
    cfg = DatabaseConfig(
        server="srv",
        database="db",
        driver="ODBC Driver 17 for SQL Server",
        trusted_connection=False,
        username="ocr_user",
        password="s3cr3t",
    )
    url = build_connection_url(cfg)
    assert url.username == "ocr_user"
    assert "trusted_connection" not in url.query


# ── create_db_engine ──────────────────────────────────────────────────────────


def test_create_db_engine_calls_sqlalchemy(mocker):
    """create_db_engine must pass the right keyword arguments to SQLAlchemy."""
    mock_sa = mocker.patch("pipeline.db.sa_create_engine", return_value=MagicMock())
    cfg = DatabaseConfig()
    create_db_engine(cfg)
    mock_sa.assert_called_once()
    _, kwargs = mock_sa.call_args
    assert kwargs["pool_size"] == 5
    assert kwargs["pool_pre_ping"] is True


# ── init_db ───────────────────────────────────────────────────────────────────


def test_init_db_creates_tables(sqlite_engine):
    """init_db should be idempotent — calling it twice must not raise."""
    # Tables were already created by the sqlite_engine fixture
    init_db(sqlite_engine)  # second call is safe
    inspector = inspect(sqlite_engine)
    assert "ocr_images" in inspector.get_table_names()
    assert "ocr_results" in inspector.get_table_names()


# ── OCRImage model ────────────────────────────────────────────────────────────


def test_ocr_image_insert_and_query(sqlite_engine):
    with Session(sqlite_engine) as session:
        img = OCRImage(
            file_path="/scans/doc.tiff",
            file_name="doc.tiff",
            file_size_bytes=4096,
            status="pending",
            retry_count=0,
        )
        session.add(img)
        session.commit()
        loaded = session.get(OCRImage, img.id)

    assert loaded is not None
    assert loaded.file_path == "/scans/doc.tiff"
    assert loaded.status == "pending"
    assert isinstance(loaded.created_at, datetime)


def test_ocr_image_optional_fields_nullable(sqlite_engine):
    """Optional fields default to None without raising."""
    with Session(sqlite_engine) as session:
        img = OCRImage(file_path="/a.tif", file_name="a.tif", status="pending")
        session.add(img)
        session.commit()
        loaded = session.get(OCRImage, img.id)

    assert loaded.worker_id is None
    assert loaded.started_at is None
    assert loaded.error_message is None


# ── OCRResult model ───────────────────────────────────────────────────────────


def test_ocr_result_relationship(sqlite_engine):
    """OCRResult must link back to OCRImage via the relationship."""
    with Session(sqlite_engine) as session:
        img = OCRImage(file_path="/b.tiff", file_name="b.tiff", status="pending")
        session.add(img)
        session.flush()

        result = OCRResult(
            image_id=img.id,
            page_number=1,
            extracted_text="hello world",
            confidence_score=0.92,
            processing_time_ms=250,
            ocr_engine="tesseract",
        )
        session.add(result)
        session.commit()

        loaded_img = session.get(OCRImage, img.id)
        # Access relationship while the session is open to avoid DetachedInstanceError
        assert len(loaded_img.results) == 1
        assert loaded_img.results[0].extracted_text == "hello world"
        assert loaded_img.results[0].ocr_engine == "tesseract"


def test_ocr_result_unique_constraint(sqlite_engine):
    """Inserting two OCRResults for the same image/page must raise."""
    from sqlalchemy.exc import IntegrityError

    with Session(sqlite_engine) as session:
        img = OCRImage(file_path="/c.tiff", file_name="c.tiff", status="pending")
        session.add(img)
        session.flush()
        session.add(OCRResult(image_id=img.id, page_number=1, extracted_text="p1"))
        session.add(OCRResult(image_id=img.id, page_number=1, extracted_text="dup"))
        with pytest.raises(IntegrityError):
            session.commit()
