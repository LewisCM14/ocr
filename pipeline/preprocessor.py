''' Image preprocessor
~~~~~~~~~~~~~~~~~~~
Prepares scanned TIFF pages for OCR with a chain optimised for aged, low-quality document scans:
1. DPI normalisation — upscale to target DPI if the scan DPI is too low
2. Grayscale — reduce to single channel
3. Denoising — non-local means filter (gentle, preserves edges)
4. Deskewing — detect and correct page rotation via minAreaRect
5. Binarisation — Sauvola adaptive thresholding (handles uneven illumination better than global Otsu for old docs)
All steps are individually configurable via PreprocessingConfig.
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
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

        # --- 1. DPI normalisation ---
        # Always use the first value of the tuple for current_dpi
        if isinstance(original_dpi, tuple):
            current_dpi = float(original_dpi[0])
        else:
            current_dpi = float(original_dpi)
        logger.info(f'Preprocessing: current_dpi={current_dpi}, target_dpi={self.cfg.target_dpi}')

        # Sanity check: if DPI is missing or unreasonable, use default
        if not current_dpi or current_dpi < 10 or current_dpi > 1200:
            logger.warning(f'Unusual DPI detected ({current_dpi}), using default {self.cfg.default_dpi}')
            current_dpi = self.cfg.default_dpi

        scale = self.cfg.target_dpi / current_dpi
        logger.info(f'Preprocessing: scale={scale}')
        # Sanity check: avoid massive upscaling
        if scale > 5.0:
            logger.warning(f'Scale factor {scale} is too high, skipping upscaling.')
            scale = 1.0
        if scale != 1.0 and current_dpi < self.cfg.min_dpi:
            new_w = max(1, int(page.width * scale))
            new_h = max(1, int(page.height * scale))
            logger.info(f'Upscaling image from {page.width}x{page.height} to {new_w}x{new_h}')
            # Prevent upscaling to absurd sizes
            if new_w * new_h > 20_000_000:  # e.g., >20MP
                logger.error(f'Requested upscaled size {new_w}x{new_h} is too large, skipping upscaling.')
            else:
                page = page.resize((new_w, new_h), Image.LANCZOS)
                was_upscaled = True

        # --- 2. Grayscale ---
        if page.mode != 'L':
            page = page.convert('L')
        arr: np.ndarray = np.array(page, dtype=np.uint8)

        # --- 3. Denoising ---
        if self.cfg.denoise:
            logger.info('Applying denoising')
            arr = cv2.fastNlMeansDenoising(
                arr,
                h=float(self.cfg.denoise_strength),
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # --- 4. Deskewing ---
        if self.cfg.deskew:
            logger.info('Estimating skew')
            skew_angle = self._estimate_skew(arr)
            if abs(skew_angle) >= self.cfg.deskew_threshold_degrees:
                logger.info(f'Deskewing by {skew_angle:.2f} degrees')
                arr = self._rotate(arr, skew_angle)
                was_deskewed = True

        # --- 5. Binarisation ---
        bin_method = (self.cfg.binarization or 'none').lower()
        logger.info(f'Binarization method: {bin_method}')
        if bin_method == 'sauvola':
            arr = self._sauvola(arr)
        elif bin_method == 'otsu':
            arr = self._otsu(arr)
        # 'none' → pass grayscale directly to Tesseract

        return ProcessedPage(
            image=Image.fromarray(arr),
            original_dpi=original_dpi,
            was_upscaled=was_upscaled,
            was_deskewed=was_deskewed,
            skew_angle_degrees=skew_angle,
        )

    # ── Helpers ───────────────────────────────────────────────────────────
    def _get_dpi(self, image: Image.Image) -> Tuple[float, float]:
        '''Read DPI from TIFF metadata, fall back to configured default.'''
        try:
            dpi = image.info.get('dpi')
            if dpi:
                # If it's a tuple, return as is; if it's a single value, duplicate it
                if isinstance(dpi, tuple):
                    return float(dpi), float(dpi)
                else:
                    return float(dpi), float(dpi)
        except Exception:
            pass
        return float(self.cfg.default_dpi), float(self.cfg.default_dpi)

    @staticmethod
    def _estimate_skew(arr: np.ndarray) -> float:
        ''' Estimate page rotation angle using the minimum-area bounding rectangle of all foreground (dark) pixels. Returns angle in degrees; positive values indicate clockwise tilt. '''
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
        '''Rotate the image by `angle` degrees, filling new pixels with white.'''
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )

    @staticmethod
    def _sauvola(arr: np.ndarray) -> np.ndarray:
        ''' Sauvola adaptive binarisation — recommended for historical documents with uneven illumination, foxing, or bleed-through. '''
        from skimage.filters import threshold_sauvola
        threshold = threshold_sauvola(arr, window_size=25)
        binary = (arr > threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _otsu(arr: np.ndarray) -> np.ndarray:
        '''Global Otsu binarisation — faster but assumes bimodal histogram.'''
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary