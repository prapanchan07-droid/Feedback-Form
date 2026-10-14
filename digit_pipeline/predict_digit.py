"""
predict_digit.py
────────────────
Adapter that plugs into the digit_pipeline / form_merge.py CNN stub.

Public API
──────────
    predict_digit(crop_28x28)   -> tuple[str, float]
    predict_batch(crops)        -> list[tuple[str, float]]
    normalize_crop(crop_28x28)  -> np.ndarray  shape (1,28,28,1) float32

Configuration (environment variables)
──────────────────────────────────────
    DIGIT_MODEL_PATH   path to digit_cnn.keras
                       default: CNN/outputs/digit_cnn.keras
    DIGIT_MIN_INK      minimum foreground (white) pixel count before "blank"
                       default: 10
    DIGIT_MIN_COMP     minimum largest-connected-component area before "blank"
                       default: 8
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent / "CNN" / "outputs" / "digit_cnn.keras"
)
DIGIT_MODEL_PATH: str = os.environ.get("DIGIT_MODEL_PATH", _DEFAULT_MODEL_PATH)
DIGIT_MIN_INK: int    = int(os.environ.get("DIGIT_MIN_INK", "10"))
DIGIT_MIN_COMP: int   = int(os.environ.get("DIGIT_MIN_COMP", "8"))

# ── Lazy model cache ──────────────────────────────────────────────────────────
_model: Any = None   # tensorflow.keras.Model, loaded on first call


def _load_model() -> Any:
    """
    Load the Keras model once and cache it.  Importing tensorflow is deferred
    to here so the module can be imported without tensorflow installed (e.g. in
    tests that mock the model).
    """
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # noqa: PLC0415

    path = DIGIT_MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Keras digit model not found at '{path}'. "
            "Set DIGIT_MODEL_PATH or copy CNN/outputs/digit_cnn.keras to the default location."
        )
    _model = tf.keras.models.load_model(path)
    return _model


# ── Pure geometry function (testable without tensorflow) ─────────────────────

def normalize_crop(crop_28x28: np.ndarray) -> np.ndarray:
    """
    Convert a pre-binarised digit crop (output of teammate's _preprocess_box)
    into the (1, 28, 28, 1) float32 tensor expected by the Keras model.

    Accepted input forms
    ────────────────────
    • uint8   (28, 28)    — white ink on black  (0 / 255)
    • uint8   (28, 28, 1) — same, with channel dim
    • float32 (28, 28)    — already in [0, 1] (pass-through after reshape)
    • float32 (28, 28, 1) — same

    Steps (NO re-threshold, NO blur — input is already binary)
    ──────────────────────────────────────────────────────────
    1.  Squeeze to (28, 28).
    2.  Normalise to float32 [0.0, 1.0].
    3.  Tight bounding-rect crop of the ink.
    4.  Aspect-preserving resize into a 20×20 box.
    5.  Paste centred on a 28×28 black canvas.
    6.  Centre-of-mass shift (intensity-weighted, MNIST style).
    7.  Reshape to (1, 28, 28, 1).

    Returns
    ───────
    np.ndarray  shape (1, 28, 28, 1)  dtype float32  values in [0.0, 1.0]
    """
    # ── Step 1: normalise input shape and dtype ───────────────────────────────
    arr = np.squeeze(crop_28x28)                   # -> (28, 28)
    if arr.ndim != 2:
        raise ValueError(
            f"normalize_crop: expected 2-D input after squeeze, got shape {arr.shape}"
        )

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Detect uint8-encoded binary [0, 255] and rescale to [0, 1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # ── Step 2: to uint8 for OpenCV geometry ops ─────────────────────────────
    # (we only use cv2 for bounding-rect and resize; result is normalised back)
    u8 = (arr * 255.0).astype(np.uint8)

    # ── Step 3: tight bounding-rect of ink pixels ────────────────────────────
    coords = cv2.findNonZero(u8)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        digit = u8[by : by + bh, bx : bx + bw]
    else:
        digit = u8                              # blank / no ink — handled above

    dh, dw = digit.shape
    if dh == 0 or dw == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    # ── Step 4: aspect-preserving resize to 20×20 ────────────────────────────
    scale = 20.0 / max(dh, dw)
    new_w = max(1, int(round(dw * scale)))
    new_h = max(1, int(round(dh * scale)))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── Step 5: paste centred on 28×28 black canvas ──────────────────────────
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0 = (28 - new_w) // 2
    y0 = (28 - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # ── Step 6: intensity-weighted centre-of-mass shift ──────────────────────
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dx = int(round(13.5 - cx))
        dy = int(round(13.5 - cy))
        M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(
            canvas, M_trans, (28, 28),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # ── Step 7: float32 [0,1] and reshape to (1,28,28,1) ────────────────────
    out = canvas.astype(np.float32) / 255.0
    return out.reshape(1, 28, 28, 1)


# ── Blank detection ──────────────────────────────────────────────────────────

def _is_blank(
    u8: np.ndarray,
    min_ink: int = DIGIT_MIN_INK,
    min_comp: int = DIGIT_MIN_COMP,
) -> bool:
    """
    Return True if the crop contains no meaningful digit ink.

    Checks (both must hold for the crop to be non-blank):
    1. Total white pixels  ≥ min_ink
    2. Largest connected component area  ≥ min_comp
    """
    ink_pixels = int(np.count_nonzero(u8))
    if ink_pixels < min_ink:
        return True

    # Connected components on the binary image
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n_labels <= 1:                           # only background label 0
        return True

    # stats[:, cv2.CC_STAT_AREA] has area for each label (label 0 = background)
    max_comp_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return max_comp_area < min_comp


def _to_u8(crop_28x28: np.ndarray) -> np.ndarray:
    """Squeeze and convert any accepted input form to uint8 (28,28)."""
    arr = np.squeeze(crop_28x28)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


# ── Public API ────────────────────────────────────────────────────────────────

def predict_digit(
    crop_28x28: np.ndarray,
    min_ink: int = DIGIT_MIN_INK,
    min_comp: int = DIGIT_MIN_COMP,
) -> tuple[str, float]:
    """
    Main entry-point matching the teammate's contract:

        predict_digit(crop_28x28) -> (digit_str, confidence)

    Parameters
    ──────────
    crop_28x28 : np.ndarray
        Pre-binarised crop from _preprocess_box; white ink on black background.
        Accepted shapes/dtypes: uint8 (28,28), uint8 (28,28,1),
                                float32 (28,28), float32 (28,28,1).
    min_ink    : int   minimum ink-pixel count for non-blank (default: DIGIT_MIN_INK)
    min_comp   : int   minimum largest-CC area  for non-blank (default: DIGIT_MIN_COMP)

    Returns
    ───────
    digit_str  : str    "0" .. "9"  or  "" (blank / no digit)
    confidence : float  in [0.0, 1.0]  (1.0 for blank shortcut)
    """
    u8 = _to_u8(crop_28x28)

    # ── Blank check FIRST ─────────────────────────────────────────────────────
    if _is_blank(u8, min_ink=min_ink, min_comp=min_comp):
        return "", 1.0

    # ── Geometry normalisation ─────────────────────────────────────────────────
    tensor = normalize_crop(u8)                 # (1, 28, 28, 1) float32

    # ── Inference ─────────────────────────────────────────────────────────────
    model = _load_model()
    probs = model.predict(tensor, verbose=0)[0]  # (10,) float32

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return str(pred_idx), confidence


def predict_batch(
    crops: list[np.ndarray],
    min_ink: int = DIGIT_MIN_INK,
    min_comp: int = DIGIT_MIN_COMP,
) -> list[tuple[str, float]]:
    """
    Batch inference — faster than calling predict_digit in a loop because
    it groups non-blank crops into a single model.predict call.

    Parameters
    ──────────
    crops : list of ndarray   each element accepted by predict_digit

    Returns
    ───────
    list of (digit_str, confidence) in the same order as input
    """
    results: list[tuple[str, float]] = [("", 1.0)] * len(crops)

    non_blank_indices: list[int] = []
    tensors: list[np.ndarray] = []

    for i, crop in enumerate(crops):
        u8 = _to_u8(crop)
        if _is_blank(u8, min_ink=min_ink, min_comp=min_comp):
            results[i] = ("", 1.0)
        else:
            non_blank_indices.append(i)
            tensors.append(normalize_crop(u8))   # (1,28,28,1)

    if not tensors:
        return results

    # Stack into (N, 28, 28, 1)
    batch = np.concatenate(tensors, axis=0)
    model = _load_model()
    all_probs = model.predict(batch, verbose=0)  # (N, 10)

    for j, i in enumerate(non_blank_indices):
        pred_idx = int(np.argmax(all_probs[j]))
        conf     = float(all_probs[j][pred_idx])
        results[i] = (str(pred_idx), conf)

    return results
