"""Tests for pipeline/config.py."""

from __future__ import annotations

import pytest

from pipeline.config import (
    Config,
    DatabaseConfig,
    InputConfig,
    OCRConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    load_config,
)


# ── Dataclass defaults ────────────────────────────────────────────────────────


def test_database_config_defaults():
    cfg = DatabaseConfig()
    assert cfg.server == "localhost"
    assert cfg.trusted_connection is True
    assert cfg.username is None
    assert cfg.password is None


def test_input_config_defaults():
    cfg = InputConfig()
    assert cfg.extensions == [".tif", ".tiff"]
    assert cfg.recursive is True


def test_output_config_defaults():
    cfg = OutputConfig()
    assert cfg.formats == ["txt", "json"]


def test_ocr_config_defaults():
    cfg = OCRConfig()
    assert cfg.engine == "tesseract"
    assert cfg.language == "eng"
    assert cfg.default_dpi == 300
    assert cfg.tesseract_cmd is None


def test_preprocessing_config_defaults():
    cfg = PreprocessingConfig()
    assert cfg.enabled is True
    assert cfg.min_dpi == 200
    assert cfg.target_dpi == 300
    assert cfg.default_dpi == 300
    assert cfg.binarization == "sauvola"


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.num_workers == 4
    assert cfg.max_retries == 3


def test_config_top_level_defaults():
    cfg = Config()
    assert isinstance(cfg.database, DatabaseConfig)
    assert isinstance(cfg.input, InputConfig)
    assert isinstance(cfg.output, OutputConfig)
    assert isinstance(cfg.ocr, OCRConfig)
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert isinstance(cfg.pipeline, PipelineConfig)


# ── load_config ───────────────────────────────────────────────────────────────

_FULL_YAML = """\
database:
  server: "db-server"
  database: "my_db"
  driver: "ODBC Driver 18 for SQL Server"
  trusted_connection: false
  username: "ocr_user"
  password: "s3cr3t"

input:
  root_path: "/data/input"
  extensions:
    - ".tif"
  recursive: false

output:
  root_path: "/data/output"
  formats:
    - "txt"

ocr:
  engine: "tesseract"
  language: "fra"
  tesseract_config: "--oem 1 --psm 6"
  default_dpi: 200

preprocessing:
  enabled: false
  min_dpi: 150
  target_dpi: 400
  default_dpi: 150
  deskew: false
  deskew_threshold_degrees: 1.0
  denoise: false
  denoise_strength: 5
  binarization: "otsu"

pipeline:
  num_workers: 8
  batch_size: 20
  max_retries: 5
  stale_processing_minutes: 30
  log_level: "DEBUG"
  log_dir: "/logs"
"""


def test_load_config_all_sections(tmp_path):
    """All 6 section keys present — every setattr branch is True."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_FULL_YAML)

    cfg = load_config(str(cfg_file))

    assert cfg.database.server == "db-server"
    assert cfg.database.username == "ocr_user"
    assert cfg.input.root_path == "/data/input"
    assert cfg.input.recursive is False
    assert cfg.output.root_path == "/data/output"
    assert cfg.ocr.language == "fra"
    assert cfg.preprocessing.enabled is False
    assert cfg.preprocessing.min_dpi == 150
    assert cfg.pipeline.num_workers == 8


def test_load_config_empty_yaml(tmp_path):
    """Empty YAML ({}) — every setattr branch is False; all defaults used."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}\n")

    cfg = load_config(str(cfg_file))

    # All fields should be their dataclass defaults
    assert cfg.database.server == "localhost"
    assert cfg.input.recursive is True
    assert cfg.ocr.engine == "tesseract"
    assert cfg.preprocessing.binarization == "sauvola"
    assert cfg.pipeline.num_workers == 4


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
