"""Discovery module
~~~~~~~~~~~~~~~~~~~
Walks the input filesystem tree, finds all TIFF files, and registers them in the database ready for processing.
Designed to be re-run safely at any time:
- Files already in the database are skipped (no re-registration).
- New files added to the filesystem since the last discovery run are picked up.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine
from .config import InputConfig


def iter_tiff_files(cfg: InputConfig) -> Iterator[Path]:
    """Yield every TIFF file path under the configured root directory."""
    root = Path(cfg.root_path)
    extensions = {ext.lower() for ext in cfg.extensions}
    if cfg.recursive:
        yield from _iter_tiff_files_recursive(root, extensions)
    else:
        yield from _iter_tiff_files_non_recursive(root, extensions)


def _iter_tiff_files_recursive(root: Path, extensions: set[str]) -> Iterator[Path]:
    """Recursive generator for TIFF files under `root`."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if Path(filename).suffix.lower() in extensions:
                yield Path(dirpath) / filename


def _iter_tiff_files_non_recursive(root: Path, extensions: set[str]) -> Iterator[Path]:
    """Non-recursive generator for TIFF files directly under `root`."""
    for entry in os.scandir(root):
        if entry.is_file() and Path(entry.name).suffix.lower() in extensions:
            yield Path(entry.path)


def register_images(engine: Engine, cfg: InputConfig, batch_size: int = 500) -> dict:
    """
    Scan the filesystem and register all discovered TIFF files in the database.
    Returns a stats dict: {'discovered': int, 'registered': int, 'skipped': int}
    """
    stats = {"discovered": 0, "registered": 0, "skipped": 0}
    batch: list[dict] = []
    logger.info(f"Scanning '{cfg.root_path}' for TIFF images...")
    for path in iter_tiff_files(cfg):
        stats["discovered"] += 1
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        batch.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "file_size_bytes": size,
            }
        )
        if len(batch) >= batch_size:
            r, s = _insert_batch(engine, batch)
            stats["registered"] += r
            stats["skipped"] += s
            batch.clear()
            logger.info(
                f" discovered={stats['discovered']:,} "
                f"registered={stats['registered']:,} "
                f"skipped={stats['skipped']:,}"
            )
    if batch:
        r, s = _insert_batch(engine, batch)
        stats["registered"] += r
        stats["skipped"] += s
    logger.info(
        f"Discovery complete — "
        f"discovered={stats['discovered']:,}, "
        f"registered={stats['registered']:,}, "
        f"skipped (already in DB)={stats['skipped']:,}"
    )
    return stats


def _insert_batch(engine: Engine, batch: list[dict]) -> tuple[int, int]:
    """
    Insert records for files not yet in the database.
    Uses a conditional INSERT to avoid unique-constraint violations.
    Returns (registered_count, skipped_count).
    """
    registered = 0
    skipped = 0
    with engine.begin() as conn:
        dialect = conn.engine.dialect.name
        for item in batch:
            if dialect == "sqlite":
                result = conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO ocr_images
                            (file_path, file_name, file_size_bytes, status, retry_count, created_at)
                        VALUES
                            (:file_path, :file_name, :file_size_bytes, 'pending', 0, CURRENT_TIMESTAMP)
                        """
                    ),
                    item,
                )
            else:  # Assume SQL Server
                result = conn.execute(
                    text(
                        """
                        IF NOT EXISTS (
                            SELECT 1 FROM ocr_images WHERE file_path = :file_path
                        )
                        BEGIN
                            INSERT INTO ocr_images
                                (file_path, file_name, file_size_bytes, status, retry_count, created_at)
                            VALUES
                                (:file_path, :file_name, :file_size_bytes, 'pending', 0, GETUTCDATE())
                        END
                        """
                    ),
                    item,
                )
            if result.rowcount == 1:
                registered += 1
            else:
                skipped += 1
    return registered, skipped
