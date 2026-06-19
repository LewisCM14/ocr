"""Tests for pipeline/discovery.py."""

from __future__ import annotations

from unittest.mock import MagicMock


from pipeline.config import InputConfig
from pipeline.discovery import _insert_batch, iter_tiff_files, register_images
from tests.conftest import make_mock_engine


# ── iter_tiff_files ───────────────────────────────────────────────────────────


def test_iter_tiff_files_recursive_finds_nested(tmp_path):
    """Recursive walk must yield TIFF files in sub-directories."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.tif").touch()
    (tmp_path / "sub" / "b.tiff").touch()
    (tmp_path / "c.jpg").touch()  # not a TIFF — should be skipped

    cfg = InputConfig(
        root_path=str(tmp_path), extensions=[".tif", ".tiff"], recursive=True
    )
    found = {p.name for p in iter_tiff_files(cfg)}

    assert found == {"a.tif", "b.tiff"}


def test_iter_tiff_files_recursive_ignores_non_tiff(tmp_path):
    (tmp_path / "doc.pdf").touch()
    (tmp_path / "img.png").touch()

    cfg = InputConfig(
        root_path=str(tmp_path), extensions=[".tif", ".tiff"], recursive=True
    )
    assert list(iter_tiff_files(cfg)) == []


def test_iter_tiff_files_non_recursive_skips_subdirs(tmp_path):
    """Non-recursive scan must not descend into sub-directories."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "top.tif").touch()
    (tmp_path / "sub" / "nested.tiff").touch()

    cfg = InputConfig(
        root_path=str(tmp_path), extensions=[".tif", ".tiff"], recursive=False
    )
    found = [p.name for p in iter_tiff_files(cfg)]

    assert found == ["top.tif"]


def test_iter_tiff_files_non_recursive_extension_filter(tmp_path):
    """Non-recursive scan respects extension list."""
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.jpg").touch()

    cfg = InputConfig(root_path=str(tmp_path), extensions=[".tif"], recursive=False)
    found = [p.name for p in iter_tiff_files(cfg)]

    assert found == ["a.tif"]


# ── _insert_batch ─────────────────────────────────────────────────────────────


def test_insert_batch_counts_registered():
    """rowcount == 1 signals a new row was inserted."""
    engine = make_mock_engine(rowcount=1)
    batch = [{"file_path": "/a.tif", "file_name": "a.tif", "file_size_bytes": 512}]

    registered, skipped = _insert_batch(engine, batch)

    assert registered == 1
    assert skipped == 0


def test_insert_batch_counts_skipped():
    """rowcount == 0 signals the row already exists (skip)."""
    engine = make_mock_engine(rowcount=0)
    batch = [{"file_path": "/a.tif", "file_name": "a.tif", "file_size_bytes": 512}]

    registered, skipped = _insert_batch(engine, batch)

    assert registered == 0
    assert skipped == 1


def test_insert_batch_mixed():
    """Batch with both new and duplicate rows are tallied correctly."""
    mock_result_new = MagicMock()
    mock_result_new.rowcount = 1
    mock_result_dup = MagicMock()
    mock_result_dup.rowcount = 0

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = [mock_result_new, mock_result_dup]

    engine = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    batch = [
        {"file_path": "/new.tif", "file_name": "new.tif", "file_size_bytes": 100},
        {"file_path": "/dup.tif", "file_name": "dup.tif", "file_size_bytes": 200},
    ]
    registered, skipped = _insert_batch(engine, batch)

    assert registered == 1
    assert skipped == 1


# ── register_images ───────────────────────────────────────────────────────────


def test_register_images_basic(tmp_path, mocker):
    """Discovered files are forwarded to _insert_batch and stats accumulated."""
    (tmp_path / "a.tif").write_bytes(b"fake")
    (tmp_path / "b.tiff").write_bytes(b"fake")

    cfg = InputConfig(
        root_path=str(tmp_path), extensions=[".tif", ".tiff"], recursive=False
    )
    mocker.patch("pipeline.discovery._insert_batch", return_value=(2, 0))
    engine = MagicMock()

    stats = register_images(engine, cfg, batch_size=500)

    assert stats["discovered"] == 2
    assert stats["registered"] == 2
    assert stats["skipped"] == 0


def test_register_images_batch_flush(tmp_path, mocker):
    """When discovered > batch_size, _insert_batch is flushed mid-loop."""
    for i in range(3):
        (tmp_path / f"f{i}.tif").write_bytes(b"x")

    cfg = InputConfig(root_path=str(tmp_path), extensions=[".tif"], recursive=False)
    mock_insert = mocker.patch("pipeline.discovery._insert_batch", return_value=(1, 0))
    engine = MagicMock()

    stats = register_images(engine, cfg, batch_size=2)  # flush at 2, remainder 1

    # Called twice: mid-loop flush + final flush
    assert mock_insert.call_count == 2
    assert stats["discovered"] == 3


def test_register_images_batch_exactly_divisible(tmp_path, mocker):
    """When file count == batch_size exactly, the remainder batch is empty
    (False branch of `if batch:`) and no final _insert_batch call is made."""
    for i in range(2):
        (tmp_path / f"f{i}.tif").write_bytes(b"x")

    cfg = InputConfig(root_path=str(tmp_path), extensions=[".tif"], recursive=False)
    mock_insert = mocker.patch("pipeline.discovery._insert_batch", return_value=(1, 0))
    engine = MagicMock()

    stats = register_images(engine, cfg, batch_size=2)  # 2 files, batch_size=2

    # Only the mid-loop flush call; no remainder → if batch: is False
    assert mock_insert.call_count == 1
    assert stats["discovered"] == 2


def test_register_images_stat_error(tmp_path, mocker):
    """OSError from path.stat() is caught and file_size_bytes is set to None."""
    (tmp_path / "a.tif").write_bytes(b"x")

    cfg = InputConfig(root_path=str(tmp_path), extensions=[".tif"], recursive=False)
    mocker.patch("pipeline.discovery._insert_batch", return_value=(1, 0))
    mocker.patch("pathlib.Path.stat", side_effect=OSError("permission denied"))
    engine = MagicMock()

    # Should not raise; size is None but registration proceeds
    stats = register_images(engine, cfg, batch_size=500)
    assert stats["discovered"] == 1
