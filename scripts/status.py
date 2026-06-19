"""
scripts/status.py
~~~~~~~~~~~~~~~~~
Print a live summary of pipeline progress.

Usage
-----
    python scripts/status.py
    python scripts/status.py --config path/to/config.yaml --watch
    python scripts/status.py --failed        # list permanently failed images
    python scripts/status.py --reset-failed  # reset failed images back to pending
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config
from pipeline.db import create_db_engine


def _get_counts(engine) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, COUNT(*) FROM ocr_images GROUP BY status")
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def _print_summary(counts: dict) -> None:
    total = sum(counts.values())
    done = counts.get("complete", 0)
    pct = (done / total * 100) if total else 0.0

    click.echo(
        f"\n{'─' * 50}\n"
        f"  Total        : {total:>10,}\n"
        f"  Complete     : {done:>10,}  ({pct:.1f}%)\n"
        f"  Pending      : {counts.get('pending', 0):>10,}\n"
        f"  Processing   : {counts.get('processing', 0):>10,}\n"
        f"  Failed       : {counts.get('failed', 0):>10,}\n"
        f"{'─' * 50}"
    )


@click.command()
@click.option("--config", default="config.yaml", show_default=True)
@click.option(
    "--watch",
    is_flag=True,
    default=False,
    help="Refresh the summary every 30 seconds.",
)
@click.option(
    "--failed",
    "show_failed",
    is_flag=True,
    default=False,
    help="List all permanently failed images.",
)
@click.option(
    "--reset-failed",
    is_flag=True,
    default=False,
    help="Reset all failed images back to pending so they will be retried.",
)
def main(config: str, watch: bool, show_failed: bool, reset_failed: bool) -> None:
    """Display OCR pipeline status."""
    cfg = load_config(config)
    engine = create_db_engine(cfg.database)

    if reset_failed:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE ocr_images
                    SET status = 'pending', retry_count = 0, error_message = NULL
                    WHERE status = 'failed'
                    """
                )
            )
        click.echo(f"Reset {result.rowcount:,} failed images back to 'pending'.")
        return

    if show_failed:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT file_path, retry_count, error_message
                    FROM ocr_images
                    WHERE status = 'failed'
                    ORDER BY file_path
                    """
                )
            ).fetchall()
        if not rows:
            click.echo("No permanently failed images.")
        else:
            click.echo(f"\n{len(rows):,} permanently failed image(s):\n")
            for row in rows:
                click.echo(f"  {row[0]}")
                click.echo(f"    retries={row[1]}  error={row[2]}\n")
        return

    if watch:
        try:
            while True:
                click.clear()
                _print_summary(_get_counts(engine))
                time.sleep(30)
        except KeyboardInterrupt:
            pass
    else:
        _print_summary(_get_counts(engine))


if __name__ == "__main__":
    main()
