from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import create_engine as sa_create_engine
from .config import DatabaseConfig


class Base(DeclarativeBase):
    pass


class OCRImage(Base):
    __tablename__ = "ocr_images"
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    results: Mapped[list["OCRResult"]] = relationship(
        "OCRResult", back_populates="image", cascade="all, delete-orphan"
    )


class OCRResult(Base):
    __tablename__ = "ocr_results"
    __table_args__ = (
        UniqueConstraint("image_id", "page_number", name="UQ_ocr_results_image_page"),
    )
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    image_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ocr_images.id"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    image: Mapped[OCRImage] = relationship("OCRImage", back_populates="results")


def build_connection_url(cfg: DatabaseConfig) -> str | URL:
    driver = (cfg.driver or "").lower()
    if driver == "sqlite":
        db_path = cfg.database or ""
        # Use three slashes for relative paths, four for absolute paths
        is_absolute_unix = db_path.startswith("/")
        is_windows_abs = len(db_path) >= 3 and db_path[1:3] == ":/"
        if is_absolute_unix or is_windows_abs:
            # Absolute paths require an extra slash: sqlite:////absolute/path
            return f"sqlite:////{db_path}"
        return f"sqlite:///{db_path}"

    query: dict[str, str] = {"driver": cfg.driver}
    if getattr(cfg, "trusted_connection", False):
        query["trusted_connection"] = "yes"
    return URL.create(
        drivername="mssql+pyodbc",
        username=None if cfg.trusted_connection else cfg.username,
        password=None if cfg.trusted_connection else cfg.password,
        host=cfg.server,
        database=cfg.database,
        query=query,
    )


def create_db_engine(cfg: DatabaseConfig) -> Engine:
    url = build_connection_url(cfg)
    # Only use pooling params for SQL Server
    if isinstance(url, str) and url.startswith("sqlite"):
        return sa_create_engine(url, future=True)
    return sa_create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_timeout=30,
        future=True,
    )


def init_db(engine: Engine) -> None:
    """Create tables that do not yet exist. Safe to call multiple times."""
    Base.metadata.create_all(engine)
