"""Tests for pipeline/manager.py."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock


from pipeline.config import (
    Config,
    DatabaseConfig,
    InputConfig,
    OCRConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from pipeline.manager import _monitor, run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_cfg(tmp_path, num_workers=1):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return Config(
        database=DatabaseConfig(),
        input=InputConfig(root_path=str(tmp_path / "input")),
        output=OutputConfig(root_path=str(tmp_path / "output")),
        ocr=OCRConfig(engine="tesseract"),
        preprocessing=PreprocessingConfig(),
        pipeline=PipelineConfig(
            num_workers=num_workers,
            log_dir=str(logs),
        ),
    )


# ── _monitor ──────────────────────────────────────────────────────────────────


def _one_shot_stop_event():
    """Returns a mock stop_event that lets the while-loop body execute once."""
    stop = MagicMock()
    stop.is_set.side_effect = [False, True]
    stop.wait.return_value = None
    return stop


def test_monitor_logs_progress_with_data():
    rows = [("complete", 5), ("pending", 3)]
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    engine = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    stop = _one_shot_stop_event()
    _monitor(engine, stop, interval_seconds=0)  # no sleep

    stop.wait.assert_called_once_with(0)


def test_monitor_handles_zero_total():
    """Empty DB (total == 0) must not cause a ZeroDivisionError."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    engine = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    stop = _one_shot_stop_event()
    _monitor(engine, stop, interval_seconds=0)  # must not raise


def test_monitor_logs_warning_on_db_exception():
    """A DB exception inside the loop body is caught and logged, loop continues."""
    engine = MagicMock()
    engine.connect.side_effect = Exception("connection lost")

    stop = _one_shot_stop_event()
    _monitor(engine, stop, interval_seconds=0)  # must not raise


# ── run_pipeline ──────────────────────────────────────────────────────────────


def _setup_run_pipeline_mocks(mocker, tmp_path, cfg):
    mocker.patch("pipeline.manager.load_config", return_value=cfg)
    mocker.patch("pipeline.manager.create_db_engine", return_value=MagicMock())
    mocker.patch("pipeline.manager.logger")

    # Build a mock executor whose submit() returns a future that succeeds
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = None

    mock_pool = MagicMock()
    mock_pool.submit.return_value = mock_future

    mock_executor_cls = MagicMock()
    mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
    mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

    mocker.patch("pipeline.manager.ProcessPoolExecutor", mock_executor_cls)
    mocker.patch("pipeline.manager.as_completed", return_value=[mock_future])

    return mock_executor_cls, mock_future


def test_run_pipeline_uses_config_num_workers(tmp_path, mocker):
    """num_workers=None → uses cfg.pipeline.num_workers from config."""
    cfg = _make_cfg(tmp_path, num_workers=2)
    mock_exec, _ = _setup_run_pipeline_mocks(mocker, tmp_path, cfg)

    run_pipeline("dummy.yaml", num_workers=None)

    mock_exec.assert_called_once_with(max_workers=2)


def test_run_pipeline_overrides_num_workers(tmp_path, mocker):
    """Explicit num_workers arg overrides the config value."""
    cfg = _make_cfg(tmp_path, num_workers=4)
    mock_exec, _ = _setup_run_pipeline_mocks(mocker, tmp_path, cfg)

    run_pipeline("dummy.yaml", num_workers=1)

    mock_exec.assert_called_once_with(max_workers=1)


def test_run_pipeline_logs_worker_exception(tmp_path, mocker):
    """A future that raises must be caught and logged — pipeline still finishes."""
    cfg = _make_cfg(tmp_path, num_workers=1)
    mocker.patch("pipeline.manager.load_config", return_value=cfg)
    mocker.patch("pipeline.manager.create_db_engine", return_value=MagicMock())
    mock_logger = mocker.patch("pipeline.manager.logger")

    crash_future = MagicMock(spec=Future)
    crash_future.result.side_effect = RuntimeError("worker exploded")

    mock_pool = MagicMock()
    mock_pool.submit.return_value = crash_future

    mock_exec_cls = MagicMock()
    mock_exec_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
    mock_exec_cls.return_value.__exit__ = MagicMock(return_value=False)

    mocker.patch("pipeline.manager.ProcessPoolExecutor", mock_exec_cls)
    mocker.patch("pipeline.manager.as_completed", return_value=[crash_future])

    run_pipeline("dummy.yaml", num_workers=1)  # must not raise

    # logger.error should have been called with the failure message
    assert mock_logger.error.called
