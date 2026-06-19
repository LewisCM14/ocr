from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatabaseConfig:
    server: str = "localhost"
    database: str = "ocr_pipeline"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class InputConfig:
    root_path: str = ""
    extensions: List[str] = field(default_factory=lambda: [".tif", ".tiff"])
    recursive: bool = True


@dataclass
class OutputConfig:
    root_path: str = ""
    formats: List[str] = field(default_factory=lambda: ["txt", "json"])


@dataclass
class OCRConfig:
    engine: str = "tesseract"
    language: str = "eng"
    tesseract_config: str = "--oem 1 --psm 3"
    tesseract_cmd: Optional[str] = None
    default_dpi: int = 300


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    min_dpi: int = 200
    target_dpi: int = 300
    default_dpi: int = 300  # assumed DPI when image metadata is absent/invalid
    deskew: bool = True
    deskew_threshold_degrees: float = 0.5
    denoise: bool = True
    denoise_strength: int = 10
    binarization: str = "sauvola"  # sauvola | otsu | none


@dataclass
class PipelineConfig:
    num_workers: int = 4
    batch_size: int = 10
    max_retries: int = 3
    stale_processing_minutes: int = 60
    log_level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    config = Config()
    section_map = {
        "database": (DatabaseConfig, "database"),
        "input": (InputConfig, "input"),
        "output": (OutputConfig, "output"),
        "ocr": (OCRConfig, "ocr"),
        "preprocessing": (PreprocessingConfig, "preprocessing"),
        "pipeline": (PipelineConfig, "pipeline"),
    }
    for key, (cls, attr) in section_map.items():
        if key in data:
            setattr(config, attr, cls(**data[key]))
    return config
