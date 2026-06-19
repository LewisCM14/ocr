"""
Pipeline manager
~~~~~~~~~~~~~~~~
Launches N worker processes via ProcessPoolExecutor and monitors overall
progress by polling the database on a background thread.

On Windows, multiprocessing uses the 'spawn' start method, so each worker
process imports this module fresh — the if __name__ == '__main__' guard in
the CLI scripts is therefore mandatory.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from .config import load_config
from .db import create_db_engine
from .worker import worker_loop


# ── Progress monitor ─────────────────────────────────────────────────────────


def _monitor(engine, stop_event: threading.Event, interval_seconds: int = 30) -> None:
    """Background thread: logs pipeline progress every `interval_seconds`."""
    while not stop_event.is_set():
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT status, COUNT(*) AS cnt
                        FROM ocr_images
                        GROUP BY status
                        """
                    )
                ).fetchall()
            counts = {r[0]: r[1] for r in rows}
            total = sum(counts.values())
            done = counts.get("complete", 0)
            pct = (done / total * 100) if total else 0
            logger.info(
                f"Progress — "
                f"total={total:,} | "
                f"complete={done:,} ({pct:.1f}%) | "
                f"pending={counts.get('pending', 0):,} | "
                f"processing={counts.get('processing', 0):,} | "
                f"failed={counts.get('failed', 0):,}"
            )
        except Exception as exc:
            logger.warning(f"Progress monitor error: {exc}")
        stop_event.wait(interval_seconds)


# ── Entry point ───────────────────────────────────────────────────────────────


def run_pipeline(config_path: str, num_workers: int | None = None) -> None:
    """
    Start the full processing pipeline.

    Parameters
    ----------
    config_path : str
        Absolute path to config.yaml.
    num_workers : int | None
        Override the number of workers from config.  None = use config value.
    """
    cfg = load_config(config_path)
    workers = num_workers if num_workers is not None else cfg.pipeline.num_workers

    # Manager-level logging (workers write to their own log files)
    log_dir = Path(cfg.pipeline.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "manager.log",
        rotation="100 MB",
        retention="30 days",
        level=cfg.pipeline.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.info(f"Starting pipeline with {workers} worker process(es)")
    logger.info(f"  Input  : {cfg.input.root_path}")
    logger.info(f"  Output : {cfg.output.root_path}")
    logger.info(f"  Engine : {cfg.ocr.engine}")

    engine = create_db_engine(cfg.database)

    # Start progress monitor thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor, args=(engine, stop_monitor), daemon=True
    )
    monitor_thread.start()

    t_start = time.monotonic()

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(worker_loop, config_path): worker_idx
                for worker_idx in range(workers)
            }
            for future in as_completed(futures):
                worker_idx = futures[future]
                try:
                    future.result()
                    logger.info(f"Worker {worker_idx} exited cleanly")
                except Exception as exc:
                    logger.error(f"Worker {worker_idx} raised an exception: {exc}")
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=5)

    elapsed = time.monotonic() - t_start
    logger.info(f"Pipeline finished in {elapsed / 3600:.2f} hours")
