"""
Image preprocessor
~~~~~~~~~~~~~~~~~~
Prepares scanned TIFF pages for OCR with a chain optimised for aged,
low-quality document scans:

  1. DPI normalisation  — upscale to target DPI if the scan DPI is too low
  2. Grayscale          — reduce to single channel
  3. Denoising          — non-local means filter (gentle, preserves edges)
  4. Deskewing          — detect and correct page rotation via minAreaRect
  5. Binarisation       — Sauvola adaptive thresholding (handles uneven
                          illumination better than global Otsu for old docs)

All steps are individually configurable via PreprocessingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PreprocessingConfig


@dataclass
class ProcessedPage:
    image: Image.Image
    original_dpi: Tuple[float, float]
    was_upscaled: bool
    was_deskewed: bool
    skew_angle_degrees: float


class ImagePreprocessor:
    def __init__(self, cfg: PreprocessingConfig) -> None:
        self.cfg = cfg

    def process_page(self, page: Image.Image) -> ProcessedPage:
        original_dpi = self._get_dpi(page)
        was_upscaled = False
        was_deskewed = False
        skew_angle = 0.0

        # ── 1. DPI normalisation ──────────────────────────────────────────
        current_dpi = original_dpi[0]
        if current_dpi < self.cfg.min_dpi:
            page = self._upscale(page, current_dpi)
            was_upscaled = True

        # ── 2. Grayscale ──────────────────────────────────────────────────
        if page.mode != "L":
            page = page.convert("L")

        arr: np.ndarray = np.array(page, dtype=np.uint8)

        # ── 3. Denoising ──────────────────────────────────────────────────
        if self.cfg.denoise:
            arr = cv2.fastNlMeansDenoising(
                arr,
                h=float(self.cfg.denoise_strength),
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # ── 4. Deskewing ──────────────────────────────────────────────────
        if self.cfg.deskew:
            skew_angle = self._estimate_skew(arr)
            if abs(skew_angle) >= self.cfg.deskew_threshold_degrees:
                arr = self._rotate(arr, skew_angle)
                was_deskewed = True

        # ── 5. Binarisation ───────────────────────────────────────────────
        if self.cfg.binarization == "sauvola":
            arr = self._sauvola(arr)
        elif self.cfg.binarization == "otsu":
            arr = self._otsu(arr)
        # "none" → pass grayscale directly to Tesseract

        return ProcessedPage(
            image=Image.fromarray(arr),
            original_dpi=original_dpi,
            was_upscaled=was_upscaled,
            was_deskewed=was_deskewed,
            skew_angle_degrees=skew_angle,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_dpi(self, image: Image.Image) -> Tuple[float, float]:
        """Read DPI from TIFF metadata, fall back to configured default."""
        try:
            dpi = image.info.get("dpi")
            if dpi and dpi[0] > 0:
                return float(dpi[0]), float(dpi[1])
        except Exception:
            pass
        return float(self.cfg.default_dpi), float(self.cfg.default_dpi)

    def _upscale(self, image: Image.Image, current_dpi: float) -> Image.Image:
        scale = self.cfg.target_dpi / current_dpi
        new_w = max(1, int(image.width * scale))
        new_h = max(1, int(image.height * scale))
        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _estimate_skew(arr: np.ndarray) -> float:
        """
        Estimate page rotation angle using the minimum-area bounding rectangle
        of all foreground (dark) pixels.  Returns angle in degrees; positive
        values indicate clockwise tilt.
        """
        # Invert so text pixels are white (required for threshold)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 5:
            return 0.0

        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]

        # cv2.minAreaRect returns angles in [-90, 0).
        # Adjust to the range (-45, 45] so that near-horizontal pages are ~0°.
        if angle < -45.0:
            angle += 90.0

        return -angle  # negate: positive = clockwise rotation needed

    @staticmethod
    def _rotate(arr: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by `angle` degrees, filling new pixels with white."""
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            arr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _sauvola(arr: np.ndarray) -> np.ndarray:
        """
        Sauvola adaptive binarisation — recommended for historical documents
        with uneven illumination, foxing, or bleed-through.
        """
        from skimage.filters import threshold_sauvola

        # Window size should be large enough to span a character (≥15 pixels).
        # 25 works well for 300 DPI scans.
        threshold = threshold_sauvola(arr, window_size=25)
        # Dark pixels (text) are below threshold → map to 0 (black)
        binary = (arr > threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _otsu(arr: np.ndarray) -> np.ndarray:
        """Global Otsu binarisation — faster but assumes bimodal histogram."""
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
