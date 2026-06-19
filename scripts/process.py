"""
scripts/process.py
~~~~~~~~~~~~~~~~~~
Start the OCR processing pipeline.

Usage
-----
    python scripts/process.py
    python scripts/process.py --config path/to/config.yaml --workers 8

The `if __name__ == '__main__'` guard is REQUIRED on Windows because
multiprocessing uses the 'spawn' start method, which re-imports this module
in each worker process.  Without the guard, each worker would recursively
spawn more workers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config
from pipeline.db import create_db_engine, init_db
from pipeline.manager import run_pipeline


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Number of worker processes (overrides pipeline.num_workers in config).",
)
def main(config: str, workers: int | None) -> None:
    """Run the OCR pipeline over all pending images."""
    cfg = load_config(config)
    effective_workers = workers if workers is not None else cfg.pipeline.num_workers

    click.echo(
        f"Starting OCR pipeline\n"
        f"  Config    : {config}\n"
        f"  Workers   : {effective_workers}\n"
        f"  Engine    : {cfg.ocr.engine}\n"
        f"  Input     : {cfg.input.root_path}\n"
        f"  Output    : {cfg.output.root_path}\n"
    )

    # Ensure schema exists (safe to run multiple times)
    engine = create_db_engine(cfg.database)
    init_db(engine)
    engine.dispose()  # close manager-side connections before forking workers

    run_pipeline(config_path=config, num_workers=effective_workers)


if __name__ == "__main__":
    main()
