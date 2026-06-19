"""Tests for scripts/discover.py, scripts/process.py, and scripts/status.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.orm import Session

from pipeline.config import (
    Config,
    DatabaseConfig,
    InputConfig,
    OCRConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from pipeline.db import OCRImage

# Import the Click commands under test
from scripts.discover import main as discover_main
from scripts.process import main as process_main
from scripts.status import _get_counts, _print_summary, main as status_main


# ── Shared helpers ────────────────────────────────────────────────────────────


def _default_cfg(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return Config(
        database=DatabaseConfig(),
        input=InputConfig(root_path=str(tmp_path / "input")),
        output=OutputConfig(root_path=str(tmp_path / "output")),
        ocr=OCRConfig(engine="tesseract"),
        preprocessing=PreprocessingConfig(),
        pipeline=PipelineConfig(num_workers=2, log_dir=str(logs)),
    )


# ── scripts/discover.py ───────────────────────────────────────────────────────


def test_discover_script_success(tmp_path, mocker):
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.discover.load_config", return_value=cfg)
    mock_engine = MagicMock()
    mocker.patch("scripts.discover.create_db_engine", return_value=mock_engine)
    mocker.patch("scripts.discover.init_db")
    mocker.patch(
        "scripts.discover.register_images",
        return_value={"discovered": 10, "registered": 8, "skipped": 2},
    )

    runner = CliRunner()
    result = runner.invoke(
        discover_main, ["--config", "dummy.yaml", "--batch-size", "100"]
    )

    assert result.exit_code == 0
    assert "Discovered" in result.output
    assert "8" in result.output
    assert "2" in result.output


# ── scripts/process.py ────────────────────────────────────────────────────────


def test_process_script_with_explicit_workers(tmp_path, mocker):
    """--workers flag overrides the config value."""
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.process.load_config", return_value=cfg)
    mock_engine = MagicMock()
    mocker.patch("scripts.process.create_db_engine", return_value=mock_engine)
    mocker.patch("scripts.process.init_db")
    mock_run = mocker.patch("scripts.process.run_pipeline")

    runner = CliRunner()
    result = runner.invoke(process_main, ["--config", "dummy.yaml", "--workers", "3"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(config_path="dummy.yaml", num_workers=3)


def test_process_script_default_workers(tmp_path, mocker):
    """Omitting --workers passes num_workers=None → run_pipeline uses config."""
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.process.load_config", return_value=cfg)
    mock_engine = MagicMock()
    mocker.patch("scripts.process.create_db_engine", return_value=mock_engine)
    mocker.patch("scripts.process.init_db")
    mock_run = mocker.patch("scripts.process.run_pipeline")

    runner = CliRunner()
    result = runner.invoke(process_main, ["--config", "dummy.yaml"])

    assert result.exit_code == 0
    # When --workers is omitted, effective_workers = cfg.pipeline.num_workers (2)
    mock_run.assert_called_once_with(config_path="dummy.yaml", num_workers=2)


# ── scripts/status.py — helpers ───────────────────────────────────────────────


def test_get_counts_returns_correct_totals(sqlite_engine):
    with Session(sqlite_engine) as s:
        s.add(OCRImage(file_path="/a.tif", file_name="a.tif", status="complete"))
        s.add(OCRImage(file_path="/b.tif", file_name="b.tif", status="pending"))
        s.add(OCRImage(file_path="/c.tif", file_name="c.tif", status="pending"))
        s.commit()

    counts = _get_counts(sqlite_engine)
    assert counts == {"complete": 1, "pending": 2}


def test_print_summary_with_data(capsys):
    _print_summary({"complete": 3, "pending": 1, "failed": 1})
    out = capsys.readouterr().out
    assert "3" in out
    assert "60.0%" in out


def test_print_summary_empty_db(capsys):
    """Total == 0 must not cause ZeroDivisionError; pct reported as 0.0%."""
    _print_summary({})
    out = capsys.readouterr().out
    assert "0.0%" in out


# ── scripts/status.py — CLI ───────────────────────────────────────────────────


def test_status_script_basic_summary(tmp_path, mocker, sqlite_engine):
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.status.load_config", return_value=cfg)
    mocker.patch("scripts.status.create_db_engine", return_value=sqlite_engine)

    runner = CliRunner()
    result = runner.invoke(status_main, ["--config", "dummy.yaml"])

    assert result.exit_code == 0
    assert "Total" in result.output


def test_status_script_reset_failed(tmp_path, mocker, sqlite_engine):
    """--reset-failed resets all failed rows and returns early."""
    with Session(sqlite_engine) as s:
        s.add(
            OCRImage(
                file_path="/f.tif", file_name="f.tif", status="failed", retry_count=3
            )
        )
        s.commit()

    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.status.load_config", return_value=cfg)
    mocker.patch("scripts.status.create_db_engine", return_value=sqlite_engine)

    runner = CliRunner()
    result = runner.invoke(status_main, ["--config", "dummy.yaml", "--reset-failed"])

    assert result.exit_code == 0
    assert "Reset" in result.output

    # Confirm DB state changed
    with Session(sqlite_engine) as s:
        img = s.execute(
            text("SELECT status FROM ocr_images WHERE file_path = '/f.tif'")
        ).fetchone()
    assert img[0] == "pending"


def test_status_script_show_failed_no_rows(tmp_path, mocker, sqlite_engine):
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.status.load_config", return_value=cfg)
    mocker.patch("scripts.status.create_db_engine", return_value=sqlite_engine)

    runner = CliRunner()
    result = runner.invoke(status_main, ["--config", "dummy.yaml", "--failed"])

    assert result.exit_code == 0
    assert "No permanently failed" in result.output


def test_status_script_show_failed_with_rows(tmp_path, mocker, sqlite_engine):
    with Session(sqlite_engine) as s:
        s.add(
            OCRImage(
                file_path="/bad.tif",
                file_name="bad.tif",
                status="failed",
                retry_count=3,
                error_message="disk full",
            )
        )
        s.commit()

    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.status.load_config", return_value=cfg)
    mocker.patch("scripts.status.create_db_engine", return_value=sqlite_engine)

    runner = CliRunner()
    result = runner.invoke(status_main, ["--config", "dummy.yaml", "--failed"])

    assert result.exit_code == 0
    assert "bad.tif" in result.output
    assert "disk full" in result.output


def test_status_script_watch_mode(tmp_path, mocker, sqlite_engine):
    """--watch enters the refresh loop; KeyboardInterrupt exits cleanly."""
    cfg = _default_cfg(tmp_path)
    mocker.patch("scripts.status.load_config", return_value=cfg)
    mocker.patch("scripts.status.create_db_engine", return_value=sqlite_engine)
    mocker.patch("scripts.status.time.sleep", side_effect=KeyboardInterrupt)
    mocker.patch("click.clear")

    runner = CliRunner()
    result = runner.invoke(status_main, ["--config", "dummy.yaml", "--watch"])

    assert result.exit_code == 0


def test_run_modules_as_main_covers_cli_guards(tmp_path, sqlite_engine, monkeypatch):
    """Execute each scripts/*.py as __main__ to cover the if __name__ == '__main__' guard.

    We inject lightweight pipeline.* modules into sys.modules so the scripts can
    run without performing heavy IO or DB side-effects.  `sys.argv` is also
    set to avoid Click parsing pytest's CLI.
    """
    import runpy
    import sys
    import types
    from contextlib import contextmanager

    cfg = _default_cfg(tmp_path)

    @contextmanager
    def _dummy_conn():
        class DummyResult:
            def fetchall(self):
                return []

        class DummyConn:
            def execute(self, *args, **kwargs):
                return DummyResult()

        yield DummyConn()

    fake_engine = types.SimpleNamespace()
    fake_engine.connect = lambda: _dummy_conn()
    fake_engine.begin = lambda: _dummy_conn()
    fake_engine.dispose = lambda: None

    # pipeline.config
    cfg_mod = types.ModuleType("pipeline.config")
    cfg_mod.load_config = lambda path: cfg
    monkeypatch.setitem(sys.modules, "pipeline.config", cfg_mod)

    # pipeline.db (returns our fake engine)
    db_mod = types.ModuleType("pipeline.db")
    db_mod.create_db_engine = lambda dbcfg: fake_engine
    db_mod.init_db = lambda engine: None
    monkeypatch.setitem(sys.modules, "pipeline.db", db_mod)

    # pipeline.discovery
    disc_mod = types.ModuleType("pipeline.discovery")
    disc_mod.register_images = lambda engine, input_cfg, batch_size=500: {
        "discovered": 0,
        "registered": 0,
        "skipped": 0,
    }
    monkeypatch.setitem(sys.modules, "pipeline.discovery", disc_mod)

    # pipeline.manager
    mgr_mod = types.ModuleType("pipeline.manager")
    mgr_mod.run_pipeline = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "pipeline.manager", mgr_mod)

    # Execute each script as a fresh __main__
    for module_name in ("scripts.discover", "scripts.process", "scripts.status"):
        sys.modules.pop(module_name, None)
        monkeypatch.setattr(sys, "argv", [module_name])
        try:
            runpy.run_module(module_name, run_name="__main__")
        except SystemExit:
            pass
