"""
OCR engine abstraction
~~~~~~~~~~~~~~~~~~~~~~
Defines a common interface so the processing pipeline is decoupled from any
specific OCR library.  Swap the engine in config.yaml; no other code changes.

Implemented engines
-------------------
  tesseract   CPU-based, production-ready today (default)

GPU-ready upgrade paths (CPU fallback also available for both)
--------------------------------------------------------------
  easyocr     — `pip install easyocr` / `conda install -c conda-forge easyocr`
                Set gpu=True in EasyOCREngine.__init__ once a GPU is available.
  paddleocr   — `pip install paddlepaddle paddleocr`
                Set use_gpu=True in PaddleOCREngine.__init__ once ready.

Both GPU engines share the same OCREngine interface, so the worker code needs
no modification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from PIL import Image

from .config import OCRConfig


if TYPE_CHECKING:
    ...  # placeholder for future type-only imports


@dataclass
class PageOCRResult:
    text: str
    confidence: float  # 0.0 – 1.0
    engine: str


class OCREngine(ABC):
    @abstractmethod
    def process_page(self, image: Image.Image) -> PageOCRResult:
        """Run OCR on a single PIL Image and return extracted text + confidence."""


# ── Tesseract ─────────────────────────────────────────────────────────────────


class TesseractEngine(OCREngine):
    """
    Wraps pytesseract.  Requires the `tesseract` binary to be on PATH
    (automatic when using the conda environment) or configured via
    ocr.tesseract_cmd in config.yaml.
    """

    def __init__(self, cfg: OCRConfig) -> None:
        import pytesseract  # imported lazily so the module loads even if not installed

        if cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

        self._pt = pytesseract
        self._lang = cfg.language
        self._config = cfg.tesseract_config

    def process_page(self, image: Image.Image) -> PageOCRResult:
        # image_to_data gives per-word confidence scores alongside the text
        data = self._pt.image_to_data(
            image,
            lang=self._lang,
            config=self._config,
            output_type=self._pt.Output.DICT,
        )
        text = self._pt.image_to_string(image, lang=self._lang, config=self._config)

        # Tesseract reports confidence 0–100; -1 means the token has no score.
        valid_confs = [
            c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0
        ]
        mean_conf = (
            (sum(valid_confs) / len(valid_confs) / 100.0) if valid_confs else 0.0
        )

        return PageOCRResult(
            text=text.strip(),
            confidence=round(mean_conf, 4),
            engine="tesseract",
        )


# ── EasyOCR (GPU upgrade path) ────────────────────────────────────────────────


class EasyOCREngine(OCREngine):
    """
    EasyOCR engine.  Set gpu=True once GPU compute is available.
    Install: conda install -c conda-forge easyocr
    """

    def __init__(self, cfg: OCRConfig, gpu: bool = False) -> None:
        import easyocr  # type: ignore[import]
        import numpy as np

        self._np = np
        languages = cfg.language.split("+")
        self._reader = easyocr.Reader(languages, gpu=gpu)

    def process_page(self, image: Image.Image) -> PageOCRResult:
        arr = self._np.array(image)
        results = self._reader.readtext(arr, detail=1)
        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        text = " ".join(texts)
        mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        return PageOCRResult(
            text=text.strip(), confidence=round(mean_conf, 4), engine="easyocr"
        )


# ── PaddleOCR (GPU upgrade path) ──────────────────────────────────────────────


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR engine.  Set use_gpu=True once GPU compute is available.
    Install: pip install paddlepaddle paddleocr
    """

    def __init__(self, cfg: OCRConfig, use_gpu: bool = False) -> None:
        from paddleocr import PaddleOCR  # type: ignore[import]
        import numpy as np

        self._np = np
        lang = cfg.language.split("+")[0]
        self._ocr = PaddleOCR(
            use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False
        )

    def process_page(self, image: Image.Image) -> PageOCRResult:
        arr = self._np.array(image)
        results = self._ocr.ocr(arr, cls=True)
        lines = results[0] if results else []
        texts = [line[1][0] for line in lines if line]
        confs = [line[1][1] for line in lines if line]
        text = "\n".join(texts)
        mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        return PageOCRResult(
            text=text.strip(), confidence=round(mean_conf, 4), engine="paddleocr"
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_engine(cfg: OCRConfig) -> OCREngine:
    if cfg.engine == "tesseract":
        return TesseractEngine(cfg)
    if cfg.engine == "easyocr":
        return EasyOCREngine(cfg, gpu=False)
    if cfg.engine == "paddleocr":
        return PaddleOCREngine(cfg, use_gpu=False)
    raise ValueError(
        f"Unknown OCR engine '{cfg.engine}'. "
        "Supported values: tesseract, easyocr, paddleocr"
    )
