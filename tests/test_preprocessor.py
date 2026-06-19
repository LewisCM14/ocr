"""Tests for pipeline/preprocessor.py — covers every branch in the chain."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from pipeline.config import PreprocessingConfig
from pipeline.preprocessor import ImagePreprocessor, ProcessedPage


# ── Helpers ───────────────────────────────────────────────────────────────────


def _gray_image(w=400, h=100, dpi=(300, 300)) -> Image.Image:
    img = Image.new("L", (w, h), color=200)
    img.info["dpi"] = dpi
    return img


def _rgb_image(w=400, h=100, dpi=(300, 300)) -> Image.Image:
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    img.info["dpi"] = dpi
    return img


def _preprocessor(
    enabled=True,
    min_dpi=200,
    target_dpi=300,
    default_dpi=300,
    deskew=False,
    deskew_threshold_degrees=0.5,
    denoise=False,
    binarization="none",
) -> ImagePreprocessor:
    return ImagePreprocessor(
        PreprocessingConfig(
            enabled=enabled,
            min_dpi=min_dpi,
            target_dpi=target_dpi,
            default_dpi=default_dpi,
            deskew=deskew,
            deskew_threshold_degrees=deskew_threshold_degrees,
            denoise=denoise,
            binarization=binarization,
        )
    )


# ── _get_dpi ──────────────────────────────────────────────────────────────────


def test_get_dpi_reads_from_metadata():
    img = _gray_image(dpi=(150, 150))
    pp = _preprocessor()
    assert pp._get_dpi(img) == (150.0, 150.0)


def test_get_dpi_fallback_when_no_info():
    """image.info has no 'dpi' key → default_dpi is returned."""
    img = Image.new("L", (100, 100))  # info = {}
    pp = _preprocessor(default_dpi=72)
    assert pp._get_dpi(img) == (72.0, 72.0)


def test_get_dpi_fallback_when_dpi_is_zero():
    """dpi[0] == 0 is treated as invalid."""
    img = Image.new("L", (100, 100))
    img.info["dpi"] = (0, 0)
    pp = _preprocessor(default_dpi=96)
    assert pp._get_dpi(img) == (96.0, 96.0)


def test_get_dpi_fallback_on_exception():
    """Any exception inside the try block falls back to default_dpi."""
    img = Image.new("L", (100, 100))
    bad_info = MagicMock()
    bad_info.get.side_effect = RuntimeError("bad metadata")
    img.info = bad_info
    pp = _preprocessor(default_dpi=120)
    assert pp._get_dpi(img) == (120.0, 120.0)


# ── _upscale ──────────────────────────────────────────────────────────────────


def test_upscale_increases_size():
    img = Image.new("L", (100, 50))
    pp = _preprocessor(target_dpi=300)
    upscaled = pp._upscale(img, current_dpi=100)
    assert upscaled.width == 300
    assert upscaled.height == 150


# ── _estimate_skew ────────────────────────────────────────────────────────────


def test_estimate_skew_returns_zero_when_too_few_coords():
    """Fewer than 5 foreground pixels → returns 0.0 without crashing."""
    blank = np.zeros((20, 20), dtype=np.uint8)  # all black; inverted = all white
    # Otsu on a uniform image may yield all-0 thresh; < 5 foreground pixels
    result = ImagePreprocessor._estimate_skew(blank)
    assert result == pytest.approx(0.0)


def test_estimate_skew_normal_path(mocker):
    """Normal path: angle in [-45, 0) is negated without adjustment."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 20:80] = 200  # bright block gives plenty of foreground pixels

    mocker.patch(
        "pipeline.preprocessor.cv2.minAreaRect",
        return_value=((50, 50), (60, 60), -10.0),
    )
    angle = ImagePreprocessor._estimate_skew(arr)
    assert angle == pytest.approx(10.0)  # -(-10.0)


def test_estimate_skew_angle_adjustment(mocker):
    """Angles < -45 receive a +90° correction before negation."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[10:90, 10:90] = 200

    mocker.patch(
        "pipeline.preprocessor.cv2.minAreaRect",
        return_value=((50, 50), (10, 80), -80.0),
    )
    angle = ImagePreprocessor._estimate_skew(arr)
    # -80 + 90 = 10 → negated → -10
    assert angle == pytest.approx(-10.0)


# ── _rotate ───────────────────────────────────────────────────────────────────


def test_rotate_preserves_shape():
    arr = np.zeros((100, 200), dtype=np.uint8)
    rotated = ImagePreprocessor._rotate(arr, 5.0)
    assert rotated.shape == arr.shape


# ── _sauvola / _otsu ──────────────────────────────────────────────────────────


def test_sauvola_returns_binary_image():
    arr = np.random.randint(100, 200, (60, 60), dtype=np.uint8)
    result = ImagePreprocessor._sauvola(arr)
    assert result.shape == arr.shape
    unique = set(np.unique(result))
    assert unique <= {0, 255}


def test_otsu_returns_binary_image():
    arr = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
    result = ImagePreprocessor._otsu(arr)
    unique = set(np.unique(result))
    assert unique <= {0, 255}


# ── process_page — DPI normalisation ─────────────────────────────────────────


def test_process_page_upscales_low_dpi_image():
    pp = _preprocessor(min_dpi=200, target_dpi=300)
    img = _gray_image(w=100, h=50, dpi=(100, 100))  # DPI < min_dpi
    result = pp.process_page(img)
    assert result.was_upscaled is True
    assert result.original_dpi == (100.0, 100.0)


def test_process_page_no_upscale_at_target_dpi():
    pp = _preprocessor(min_dpi=200, target_dpi=300)
    img = _gray_image(w=100, h=50, dpi=(300, 300))  # DPI >= min_dpi
    result = pp.process_page(img)
    assert result.was_upscaled is False


# ── process_page — grayscale conversion ──────────────────────────────────────


def test_process_page_converts_rgb_to_grayscale():
    pp = _preprocessor()
    img = _rgb_image(dpi=(300, 300))  # mode="RGB"
    result = pp.process_page(img)
    assert result.image.mode == "L"


def test_process_page_grayscale_stays_grayscale():
    pp = _preprocessor()
    img = _gray_image(dpi=(300, 300))  # already "L"
    result = pp.process_page(img)
    assert result.image.mode == "L"


# ── process_page — denoising ──────────────────────────────────────────────────


def test_process_page_with_denoising():
    pp = _preprocessor(denoise=True)
    img = _gray_image()
    result = pp.process_page(img)
    assert isinstance(result, ProcessedPage)


def test_process_page_without_denoising():
    pp = _preprocessor(denoise=False)
    img = _gray_image()
    result = pp.process_page(img)
    assert isinstance(result, ProcessedPage)


# ── process_page — deskewing ──────────────────────────────────────────────────


def test_process_page_deskew_applied_when_angle_exceeds_threshold(mocker):
    mocker.patch(
        "pipeline.preprocessor.ImagePreprocessor._estimate_skew",
        return_value=5.0,  # > 0.5° threshold
    )
    pp = _preprocessor(deskew=True, deskew_threshold_degrees=0.5)
    result = pp.process_page(_gray_image())
    assert result.was_deskewed is True
    assert result.skew_angle_degrees == pytest.approx(5.0)


def test_process_page_deskew_skipped_when_angle_below_threshold(mocker):
    mocker.patch(
        "pipeline.preprocessor.ImagePreprocessor._estimate_skew",
        return_value=0.1,  # < 0.5° threshold
    )
    pp = _preprocessor(deskew=True, deskew_threshold_degrees=0.5)
    result = pp.process_page(_gray_image())
    assert result.was_deskewed is False


def test_process_page_deskew_disabled():
    pp = _preprocessor(deskew=False)
    result = pp.process_page(_gray_image())
    assert result.was_deskewed is False
    assert result.skew_angle_degrees == pytest.approx(0.0)


# ── process_page — binarisation ───────────────────────────────────────────────


def test_process_page_sauvola_binarisation():
    pp = _preprocessor(binarization="sauvola")
    result = pp.process_page(_gray_image())
    assert isinstance(result, ProcessedPage)


def test_process_page_otsu_binarisation():
    pp = _preprocessor(binarization="otsu")
    result = pp.process_page(_gray_image())
    assert isinstance(result, ProcessedPage)


def test_process_page_no_binarisation():
    """binarization='none' skips both sauvola and otsu — returns grayscale."""
    pp = _preprocessor(binarization="none")
    result = pp.process_page(_gray_image())
    assert isinstance(result, ProcessedPage)
