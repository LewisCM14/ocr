"""
scripts/discover.py
~~~~~~~~~~~~~~~~~~~
Scan the filesystem for TIFF images and register them in the database.

Usage
-----
    python scripts/discover.py
    python scripts/discover.py --config path/to/config.yaml

Run this once before starting the processing pipeline, and again any time
new images are added to the input directory.  Already-registered files are
skipped safely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow running as `python scripts/discover.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config
from pipeline.db import create_db_engine, init_db
from pipeline.discovery import register_images


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--batch-size",
    default=500,
    show_default=True,
    help="Number of file paths inserted per database transaction.",
)
def main(config: str, batch_size: int) -> None:
    """Discover TIFF images and register them in the OCR pipeline database."""
    cfg = load_config(config)

    click.echo(f"Connecting to {cfg.database.server}/{cfg.database.database} ...")
    engine = create_db_engine(cfg.database)

    click.echo("Initialising database schema (no-op if tables already exist) ...")
    init_db(engine)

    click.echo(f"Scanning '{cfg.input.root_path}' ...")
    stats = register_images(engine, cfg.input, batch_size=batch_size)

    click.echo(
        f"\nDone.\n"
        f"  Discovered : {stats['discovered']:,}\n"
        f"  Registered : {stats['registered']:,}\n"
        f"  Skipped    : {stats['skipped']:,}  (already in database)\n"
    )


if __name__ == "__main__":
    main()
