"""Image preprocessor
~~~~~~~~~~~~~~~~~~~
Prepares scanned TIFF pages for OCR with a chain optimised for aged, low-quality document scans:
1. DPI normalisation — upscale to target DPI if the scan DPI is too low
2. Grayscale — reduce to single channel
3. Denoising — non-local means filter (gentle, preserves edges)
4. Deskewing — detect and correct page rotation via minAreaRect
5. Binarisation — Sauvola adaptive thresholding (handles uneven illumination better than global Otsu for old docs)
All steps are individually configurable via PreprocessingConfig.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from loguru import logger
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

        # Handle DPI normalization and possible upscaling
        current_dpi = self._extract_primary_dpi(original_dpi)
        logger.info(
            f"Preprocessing: current_dpi={current_dpi}, target_dpi={self.cfg.target_dpi}"
        )
        page, was_upscaled = self._maybe_upscale(page, current_dpi)

        # Grayscale
        if page.mode != "L":
            page = page.convert("L")
        arr: np.ndarray = np.array(page, dtype=np.uint8)

        # Denoising
        if self.cfg.denoise:
            logger.info("Applying denoising")
            arr = cv2.fastNlMeansDenoising(
                arr,
                h=float(self.cfg.denoise_strength),
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # Deskewing
        if self.cfg.deskew:
            logger.info("Estimating skew")
            skew_angle = self._estimate_skew(arr)
            if abs(skew_angle) >= float(self.cfg.deskew_threshold_degrees):
                logger.info(f"Deskewing by {skew_angle:.2f} degrees")
                arr = self._rotate(arr, skew_angle)
                was_deskewed = True

        # Binarisation
        bin_method = (self.cfg.binarization or "none").lower()
        logger.info(f"Binarization method: {bin_method}")
        if bin_method == "sauvola":
            arr = self._sauvola(arr)
        elif bin_method == "otsu":
            arr = self._otsu(arr)

        return ProcessedPage(
            image=Image.fromarray(arr),
            original_dpi=original_dpi,
            was_upscaled=was_upscaled,
            was_deskewed=was_deskewed,
            skew_angle_degrees=skew_angle,
        )

    # ── Helpers ───────────────────────────────────────────────────────────
    def _get_dpi(self, image: Image.Image) -> Tuple[float, float]:
        """Read DPI from TIFF metadata, fall back to configured default.

        Delegates parsing to _parse_sequence_dpi or _parse_scalar_dpi
        depending on the type of the metadata value.
        """
        default = float(self.cfg.default_dpi), float(self.cfg.default_dpi)
        try:
            dpi = image.info.get("dpi")
        except Exception:
            return default
        if dpi is None:
            return default
        if isinstance(dpi, (tuple, list)):
            return self._parse_sequence_dpi(list(dpi)) or default
        return self._parse_scalar_dpi(dpi) or default

    @staticmethod
    def _parse_sequence_dpi(seq: list[Any]) -> Optional[Tuple[float, float]]:
        """Parse a list representation of DPI into a (x, y) pair, or None if invalid."""
        if len(seq) >= 2:
            try:
                x, y = float(seq[0]), float(seq[1])
            except Exception:
                return None
            return (x, y) if x > 0 and y > 0 else None
        if len(seq) == 1:
            try:
                x = float(seq[0])
            except Exception:
                return None
            return (x, x) if x > 0 else None
        return None

    @staticmethod
    def _parse_scalar_dpi(val: Any) -> Optional[Tuple[float, float]]:
        """Parse a scalar DPI value into (x, x), or None if invalid."""
        try:
            x = float(val)
        except Exception:
            return None
        return (x, x) if x > 0 else None

    def _upscale(self, page: Image.Image, current_dpi: float) -> Image.Image:
        """Upscale `page` according to configured target DPI and return the new image.

        This helper mirrors the upscaling logic used in `process_page` so tests
        can call it directly.
        """
        scale = float(self.cfg.target_dpi) / float(current_dpi)
        if scale <= 0 or abs(scale - 1.0) < 1e-9:
            return page
        # Prevent excessive upscaling
        if scale > 5.0:
            return page
        new_w = max(1, int(page.width * scale))
        new_h = max(1, int(page.height * scale))
        if new_w * new_h > 20_000_000:
            return page
        return page.resize((new_w, new_h), Image.LANCZOS)

    def _extract_primary_dpi(self, original_dpi: Tuple[float, float]) -> float:
        """Get the primary (X) DPI value from the tuple or scalar-like pair."""
        if isinstance(original_dpi, tuple):
            return float(original_dpi[0])
        return float(original_dpi)

    def _maybe_upscale(
        self, page: Image.Image, current_dpi: float
    ) -> tuple[Image.Image, bool]:
        """Decide whether to upscale and perform the operation. Returns (page, was_upscaled)."""
        was_upscaled = False
        try:
            if not current_dpi or current_dpi < 10 or current_dpi > 1200:
                logger.warning(
                    f"Unusual DPI detected ({current_dpi}), using default {self.cfg.default_dpi}"
                )
                current_dpi = self.cfg.default_dpi
        except Exception:
            current_dpi = self.cfg.default_dpi

        scale = float(self.cfg.target_dpi) / float(current_dpi)
        logger.info(f"Preprocessing: scale={scale}")

        if scale > 5.0:
            logger.warning(f"Scale factor {scale} is too high, skipping upscaling.")
            return page, False

        if abs(scale - 1.0) > 1e-9 and current_dpi < self.cfg.min_dpi:
            new_w = max(1, int(page.width * scale))
            new_h = max(1, int(page.height * scale))
            logger.info(
                f"Upscaling image from {page.width}x{page.height} to {new_w}x{new_h}"
            )
            if new_w * new_h > 20_000_000:
                logger.error(
                    f"Requested upscaled size {new_w}x{new_h} is too large, skipping upscaling."
                )
            else:
                page = page.resize((new_w, new_h), Image.LANCZOS)
                was_upscaled = True
        return page, was_upscaled

    @staticmethod
    def _estimate_skew(arr: np.ndarray) -> float:
        """Estimate page rotation angle using the minimum-area bounding rectangle of all foreground (dark) pixels. Returns angle in degrees; positive values indicate clockwise tilt."""
        # Invert so text pixels are white (required for threshold)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.nonzero(thresh > 0))
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
            arr,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _sauvola(arr: np.ndarray) -> np.ndarray:
        """Sauvola adaptive binarisation — recommended for historical documents with uneven illumination, foxing, or bleed-through."""
        from skimage.filters import threshold_sauvola

        threshold = threshold_sauvola(arr, window_size=25)
        binary = (arr > threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _otsu(arr: np.ndarray) -> np.ndarray:
        """Global Otsu binarisation — faster but assumes bimodal histogram."""
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
