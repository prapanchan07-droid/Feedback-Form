"""
tests/test_predict_digit.py
───────────────────────────
Pytest tests for digit_pipeline/predict_digit.py.
Runs without tensorflow (the model call is mocked where needed).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Make digit_pipeline importable from tests/ ──────────────────────────────
ROOT = Path(__file__).parent.parent.parent    # Career_guidance/
sys.path.insert(0, str(ROOT))

from digit_pipeline.predict_digit import (  # noqa: E402
    _is_blank,
    _to_u8,
    normalize_crop,
    predict_batch,
    predict_digit,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_digit_crop(digit_val: int = 5, size: int = 28) -> np.ndarray:
    """Create a synthetic white digit on black background (uint8 28×28)."""
    import cv2  # noqa: PLC0415
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(
        canvas, str(digit_val),
        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA,
    )
    return canvas


def _all_black() -> np.ndarray:
    return np.zeros((28, 28), dtype=np.uint8)


def _all_white() -> np.ndarray:
    return np.full((28, 28), 255, dtype=np.uint8)


def _thin_noise(n_px: int = 5) -> np.ndarray:
    """Fewer than DIGIT_MIN_INK ink pixels — should be treated as blank."""
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[14, 10:10 + n_px] = 255
    return arr


# ── Fake model for tests that need inference ─────────────────────────────────

def _make_fake_model(pred_class: int = 7, confidence: float = 0.93) -> MagicMock:
    probs = np.zeros((1, 10), dtype=np.float32)
    probs[0, pred_class] = confidence
    probs[0, (pred_class + 1) % 10] = 1.0 - confidence

    fake = MagicMock()
    fake.predict.return_value = probs
    return fake


# ══════════════════════════════════════════════════════════════════════════════
# normalize_crop  (pure geometry — no model required)
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeCrop:

    def test_output_shape(self):
        crop = _make_digit_crop()
        out = normalize_crop(crop)
        assert out.shape == (1, 28, 28, 1), f"Expected (1,28,28,1), got {out.shape}"

    def test_output_dtype(self):
        crop = _make_digit_crop()
        out = normalize_crop(crop)
        assert out.dtype == np.float32

    def test_output_range(self):
        crop = _make_digit_crop()
        out = normalize_crop(crop)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_accepts_uint8_2d(self):
        crop = _make_digit_crop().astype(np.uint8)   # (28,28)
        out = normalize_crop(crop)
        assert out.shape == (1, 28, 28, 1)

    def test_accepts_uint8_3d(self):
        crop = np.expand_dims(_make_digit_crop(), axis=-1)  # (28,28,1)
        out = normalize_crop(crop)
        assert out.shape == (1, 28, 28, 1)

    def test_accepts_float32_2d(self):
        crop = _make_digit_crop().astype(np.float32) / 255.0
        out = normalize_crop(crop)
        assert out.shape == (1, 28, 28, 1)

    def test_accepts_float32_3d(self):
        crop = np.expand_dims(_make_digit_crop().astype(np.float32) / 255.0, -1)
        out = normalize_crop(crop)
        assert out.shape == (1, 28, 28, 1)

    def test_all_black_returns_zeros(self):
        out = normalize_crop(_all_black())
        assert out.shape == (1, 28, 28, 1)
        assert out.max() == 0.0

    def test_all_white_output_shape(self):
        # All-white: after bounding-rect crop the whole 28×28 is ink
        out = normalize_crop(_all_white())
        assert out.shape == (1, 28, 28, 1)
        assert out.max() <= 1.0

    def test_ink_pixels_preserved(self):
        crop = _make_digit_crop()
        out = normalize_crop(crop)
        # After processing some ink must survive
        assert out.max() > 0.1, "No ink pixels survived normalization"

    def test_idempotent_float_input(self):
        """float32 [0,1] input should produce the same result as uint8 input."""
        u8 = _make_digit_crop()
        f32 = u8.astype(np.float32) / 255.0
        out_u8  = normalize_crop(u8)
        out_f32 = normalize_crop(f32)
        np.testing.assert_allclose(out_u8, out_f32, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# _is_blank
# ══════════════════════════════════════════════════════════════════════════════

class TestIsBlank:

    def test_all_black_is_blank(self):
        assert _is_blank(_all_black()) is True

    def test_all_white_not_blank(self):
        assert _is_blank(_all_white()) is False

    def test_thin_noise_is_blank(self):
        # 5 isolated pixels — below default DIGIT_MIN_INK=10
        assert _is_blank(_thin_noise(5)) is True

    def test_digit_not_blank(self):
        assert _is_blank(_make_digit_crop()) is False

    def test_custom_min_ink(self):
        # 20 pixels of noise — not blank if min_ink=10, blank if min_ink=30
        arr = np.zeros((28, 28), dtype=np.uint8)
        arr[10, 5:25] = 255    # 20 pixels in one row
        assert _is_blank(arr, min_ink=10) is False
        assert _is_blank(arr, min_ink=30) is True

    def test_large_component_not_blank(self):
        # One big connected component
        arr = np.zeros((28, 28), dtype=np.uint8)
        arr[8:20, 8:20] = 255  # 12×12 = 144 pixels — definitely not blank
        assert _is_blank(arr, min_ink=10, min_comp=8) is False


# ══════════════════════════════════════════════════════════════════════════════
# predict_digit  (model mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictDigit:

    @pytest.fixture(autouse=True)
    def _mock_model(self):
        """Patch _load_model so tensorflow is never imported in tests."""
        with patch("digit_pipeline.predict_digit._load_model",
                   return_value=_make_fake_model(pred_class=7)):
            yield

    # ── Return type and format ────────────────────────────────────────────────

    def test_returns_tuple(self):
        result = predict_digit(_make_digit_crop())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_digit_str_type(self):
        digit, _ = predict_digit(_make_digit_crop())
        assert isinstance(digit, str)

    def test_confidence_is_float(self):
        _, conf = predict_digit(_make_digit_crop())
        assert isinstance(conf, float)

    def test_confidence_in_range(self):
        _, conf = predict_digit(_make_digit_crop())
        assert 0.0 <= conf <= 1.0

    def test_digit_is_valid_or_empty(self):
        digit, _ = predict_digit(_make_digit_crop())
        assert digit in {str(d) for d in range(10)} | {""}

    # ── Blank detection ───────────────────────────────────────────────────────

    def test_all_black_returns_blank(self):
        digit, conf = predict_digit(_all_black())
        assert digit == ""
        assert conf == 1.0

    def test_all_white_returns_digit(self):
        # all-white is not blank; mock returns "7"
        digit, conf = predict_digit(_all_white())
        assert digit == "7"
        assert 0.0 <= conf <= 1.0

    def test_thin_noise_returns_blank(self):
        digit, conf = predict_digit(_thin_noise(5))
        assert digit == ""

    # ── dtype / shape variants ────────────────────────────────────────────────

    def test_accepts_uint8_2d(self):
        digit, conf = predict_digit(_make_digit_crop().astype(np.uint8))
        assert isinstance(digit, str)

    def test_accepts_uint8_3d(self):
        crop = np.expand_dims(_make_digit_crop(), -1)
        digit, conf = predict_digit(crop)
        assert isinstance(digit, str)

    def test_accepts_float32_2d(self):
        crop = _make_digit_crop().astype(np.float32) / 255.0
        digit, conf = predict_digit(crop)
        assert isinstance(digit, str)

    def test_accepts_float32_3d(self):
        crop = np.expand_dims(_make_digit_crop().astype(np.float32) / 255.0, -1)
        digit, conf = predict_digit(crop)
        assert isinstance(digit, str)


# ══════════════════════════════════════════════════════════════════════════════
# predict_batch  (model mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictBatch:

    @pytest.fixture(autouse=True)
    def _mock_model(self):
        # For batch, model returns one row per non-blank crop
        fake = MagicMock()
        def _predict(batch, verbose=0):
            n = batch.shape[0]
            probs = np.zeros((n, 10), dtype=np.float32)
            probs[:, 7] = 0.93
            probs[:, 0] = 0.07
            return probs
        fake.predict.side_effect = _predict
        with patch("digit_pipeline.predict_digit._load_model", return_value=fake):
            yield

    def test_returns_list(self):
        result = predict_batch([_make_digit_crop()] * 3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_batch_matches_single(self):
        """Each batch result must equal the single-call result for that crop."""
        crops = [_make_digit_crop(i) for i in range(5)]
        batch_results  = predict_batch(crops)
        single_results = [predict_digit(c) for c in crops]
        for b, s in zip(batch_results, single_results):
            assert b[0] == s[0], f"Digit mismatch: batch={b[0]}, single={s[0]}"
            # Confidence may differ slightly only if model is non-deterministic;
            # with our mock they must be identical
            assert abs(b[1] - s[1]) < 1e-5

    def test_blank_crops_in_batch(self):
        crops = [_all_black(), _make_digit_crop(), _all_black()]
        results = predict_batch(crops)
        assert results[0] == ("", 1.0)
        assert results[2] == ("", 1.0)
        assert results[1][0] != ""

    def test_all_blank_batch(self):
        results = predict_batch([_all_black()] * 5)
        assert all(d == "" for d, _ in results)

    def test_empty_batch(self):
        results = predict_batch([])
        assert results == []

    def test_confidence_in_range_batch(self):
        results = predict_batch([_make_digit_crop(i) for i in range(10)])
        for digit, conf in results:
            assert 0.0 <= conf <= 1.0
