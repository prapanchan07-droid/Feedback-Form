"""
form_merge.py
─────────────
Task 2 – Field Merging module for the Bannari Amman handwritten-digit form.

Public API
──────────
    process_form(path)          Full end-to-end pipeline; returns result dict.
    debug_draw(img, fields)     Saves debug_fields.jpg with colour-coded fields.
    batch_process(folder, out)  Process a whole folder, write CSV.

Internal helpers (importable for unit testing):
    to_rows(boxes, y_tol)       Group detection boxes into rows.
    assign_fields(rows)         Map rows -> named fields.
    read_field(img, boxes)      Crop, preprocess, OCR each box.
    build_results(fields, ...)  Join digit strings into formatted values.
    validate(result)            Calendar / regex / range checks.

Swap-in point for the real CNN
──────────────────────────────
    Find the section marked "── CNN STUB ──" below and replace the stub
    function body with your real predict_digit implementation.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ── CNN STUB ──
# Replace this function with the real CNN when it is ready.
# The contract:
#   crop_28x28 : np.ndarray, shape (28, 28), dtype uint8
#                white digit on black background (after THRESH_BINARY_INV)
#   returns    : tuple[str, float]
#                digit_str  – one of "0".."9"  or  "" (blank / unrecognised)
#                confidence – float in [0, 1]
# ══════════════════════════════════════════════════════════════════════════════

from digit_pipeline.predict_digit import predict_digit  # noqa: E402  ← real CNN adapter

# ══════════════════════════════════════════════════════════════════════════════
# Layout constants (matches the form spec)
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_ROW_COUNTS = [8, 20, 5]   # date, mobile-row, cutoff

# Acceptable box-count ranges per row (real scans may have extras from
# the decimal-area or overlapping detections; we take leftmost N boxes).
_ROW_RANGES = {
    "date":   (6,  11, 8),   # (min, max, target)
    "mobile": (16, 24, 20),
    "cutoff": (3,  10, 5),
}

# Confidence threshold below which a digit is flagged for review
CONFIDENCE_THRESHOLD = 0.80

# YOLO inference settings
YOLO_CONF   = 0.25
YOLO_IOU    = 0.70   # default NMS IoU (lowering gives FEWER boxes, not more)
YOLO_IMGSZ  = 1280

# Template used for homography projection (Phase 1)
_TEMPLATE_BOXES_JSON = Path(__file__).parent / "boxes.json"

# Cached template boxes (loaded once on first use)
_TEMPLATE_BOXES: list[Box] | None = None


def _get_template_boxes() -> list[Box]:
    """Lazy-load and cache the 33 template boxes from boxes.json."""
    global _TEMPLATE_BOXES
    if _TEMPLATE_BOXES is None:
        _TEMPLATE_BOXES = json.loads(_TEMPLATE_BOXES_JSON.read_text())
    return _TEMPLATE_BOXES


# ── Preprocessing config (Phase 2) ────────────────────────────────────────────

_PREPROCESS_CONFIG_PATH = Path(__file__).parent / "preprocess_config.json"
_PREPROCESS_CONFIG: dict | None = None

_PREPROCESS_CONFIG_DEFAULTS: dict = {
    "version": 0,
    "ink_threshold_method":      "per_form_multiotsu",
    "ink_threshold_global":      100,
    "ink_threshold_clamp_lo":    50,
    "ink_threshold_clamp_hi":    150,
    "illumination_method":       "raw_multiotsu",
    "illumination_flatten_ksize": 51,
    "adaptive_block_size":       51,
    "adaptive_C":                10,
    "blank_cell_ink_threshold":  30,
    "min_component_area":        5,
    "max_edge_frac_for_line":    0.5,
    "min_aspect_ratio_for_line": 3.0,
    "edge_insets": {"top": 2, "bottom": 2, "left": 2, "right": 2},
    "extra_left_inset": {},
    "target_size": 28,
    "inner_size":  20,
    "center_of_mass_shift": True,
}


def _get_preprocess_config() -> dict:
    """Lazy-load preprocess_config.json; fall back to hardcoded defaults."""
    global _PREPROCESS_CONFIG
    if _PREPROCESS_CONFIG is None:
        if _PREPROCESS_CONFIG_PATH.exists():
            loaded = json.loads(_PREPROCESS_CONFIG_PATH.read_text())
            cfg = dict(_PREPROCESS_CONFIG_DEFAULTS)
            cfg.update(loaded)
            _PREPROCESS_CONFIG = cfg
        else:
            _PREPROCESS_CONFIG = dict(_PREPROCESS_CONFIG_DEFAULTS)
    return _PREPROCESS_CONFIG


def _multiotsu2_pixels(pixels: np.ndarray) -> tuple[int, int]:
    """
    Pure-numpy 3-class Otsu on a flat pixel array.
    Returns (t_ink_ghost, t_ghost_paper).
    """
    hist, _ = np.histogram(pixels.ravel(), bins=256, range=(0, 256))
    h = hist.astype(np.float64)
    total = h.sum()
    if total == 0:
        return 80, 180
    h /= total
    vals = np.arange(256, dtype=np.float64)
    p  = np.cumsum(h)
    mu = np.cumsum(h * vals)
    mu_t = mu[-1]
    best_var, best_t1, best_t2 = -1.0, 80, 180
    for t1 in range(1, 254):
        w0 = p[t1]
        if w0 <= 0 or w0 >= 1:
            continue
        m0 = mu[t1] / w0
        t2_arr = np.arange(t1 + 1, 255)
        w1 = p[t2_arr] - p[t1];  w2 = 1.0 - p[t2_arr]
        ok = (w1 > 0) & (w2 > 0)
        if not ok.any():
            continue
        t2_ok = t2_arr[ok]; w1_ok = w1[ok]; w2_ok = w2[ok]
        m1 = (mu[t2_ok] - mu[t1]) / w1_ok
        m2 = (mu_t - mu[t2_ok]) / w2_ok
        var = w0*(m0-mu_t)**2 + w1_ok*(m1-mu_t)**2 + w2_ok*(m2-mu_t)**2
        idx = int(np.argmax(var))
        if var[idx] > best_var:
            best_var = var[idx]; best_t1 = t1; best_t2 = int(t2_ok[idx])
    return best_t1, best_t2


def _compute_form_ink_threshold(img_gray: np.ndarray, all_boxes: list[Box]) -> int:
    """
    Pool pixels from all cell interiors of one form, run 3-class Otsu,
    return t1 (ink/ghost boundary) clamped to the config safe range.
    """
    cfg = _get_preprocess_config()
    t_lo     = int(cfg.get("ink_threshold_clamp_lo", 50))
    t_hi     = int(cfg.get("ink_threshold_clamp_hi", 150))
    t_global = int(cfg.get("ink_threshold_global", 100))
    inset = 3
    pixels = []
    for box in all_boxes:
        x1 = max(0, box[0] + inset);  y1 = max(0, box[1] + inset)
        x2 = min(img_gray.shape[1], box[2] - inset)
        y2 = min(img_gray.shape[0], box[3] - inset)
        if x2 > x1 and y2 > y1:
            pixels.append(img_gray[y1:y2, x1:x2].ravel())
    if not pixels:
        return t_global
    px = np.concatenate(pixels)
    t1, _ = _multiotsu2_pixels(px)
    return int(np.clip(t1, t_lo, t_hi))

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

Box   = list[int]          # [x1, y1, x2, y2]
Row   = list[Box]          # sorted left-to-right
Digit = tuple[str, float]  # (digit_str, confidence)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def project_cutoff_via_homography(
    date_boxes:   list[Box],   # 8 detected boxes in photo coords (sorted L->R)
    mobile_boxes: list[Box],   # 20 detected boxes in photo coords (sorted L->R)
    tpl_boxes:    list[Box],   # full 33 template boxes from boxes.json
    detected_cutoff: list[Box],  # all detected boxes in the cutoff row area
    iou_thr:          float = 0.50,
    inlier_ratio_min: float = 0.70,
) -> tuple[list[Box], bool, float]:
    """
    Estimate a homography from the 28 clean anchor boxes (date+mobile) and
    project the 5 template cutoff cells into photo coordinates.

    Parameters
    ----------
    date_boxes, mobile_boxes  : exactly 8 and 20 detected anchor boxes
    tpl_boxes                 : boxes.json list (33 entries); indices 0:28 are
                                anchors, 28:33 are the cutoff cells to project
    detected_cutoff           : all YOLO boxes in the cutoff row (may be 3-10)
    iou_thr                   : min IoU to prefer a detected box over projected
    inlier_ratio_min          : below this RANSAC inlier ratio -> needs_review

    Returns
    -------
    cutoff_boxes  : list of exactly 5 [x1,y1,x2,y2] in photo coords, sorted x
    needs_review  : True if homography failed or inlier ratio too low
    inlier_ratio  : float (0-1); fraction of RANSAC inliers among 28 anchors
    """
    photo_anchors = date_boxes + mobile_boxes        # 8 + 20 = 28 points
    tpl_anchors   = tpl_boxes[:28]                  # template counterparts
    tpl_cutoff_5  = tpl_boxes[28:33]                # 5 cutoff cells in template

    if len(photo_anchors) != 28 or len(tpl_anchors) != 28:
        return [], True, 0.0

    def _ctr(b: Box):
        return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]

    src = np.array([_ctr(b) for b in tpl_anchors],   dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([_ctr(b) for b in photo_anchors], dtype=np.float32).reshape(-1, 1, 2)
    # findHomography(src, dst) -> H such that dst_pt ~ H @ src_pt
    # i.e., photo_pt = H * tpl_pt  (what we want)
    # threshold=15px: generous for YOLO 1280->7680px upscale localization error
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)

    if H is None or mask is None:
        return [], True, 0.0

    inlier_ratio = float(mask.ravel().sum()) / len(mask)
    needs_review = inlier_ratio < inlier_ratio_min

    result: list[Box] = []
    for tbox in tpl_cutoff_5:
        # Project all 4 corners of the template box into photo space
        corners = np.array([
            [tbox[0], tbox[1]], [tbox[2], tbox[1]],
            [tbox[2], tbox[3]], [tbox[0], tbox[3]],
        ], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        px1, py1 = proj.min(axis=0)
        px2, py2 = proj.max(axis=0)
        proj_box: Box = [int(px1), int(py1), int(px2), int(py2)]

        # Prefer detected box if IoU >= iou_thr
        best_iou, best_det = 0.0, None
        for det in detected_cutoff:
            v = _iou(proj_box, det)
            if v > best_iou:
                best_iou, best_det = v, det

        result.append(best_det if (best_iou >= iou_thr and best_det is not None)
                      else proj_box)

    result.sort(key=lambda b: b[0])
    return result, needs_review, inlier_ratio


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-1 – Row grouping
# ══════════════════════════════════════════════════════════════════════════════

def to_rows(boxes: list[Box], y_tol: int) -> list[Row]:
    """
    Group *boxes* into rows by proximity of their y-centre.

    The algorithm uses a greedy scan:
    1. Sort boxes by y-centre.
    2. Open a new row whenever the current box's y-centre is more than
       *y_tol* pixels away from the running row's mean y-centre.
    3. Within each row, sort left-to-right by x1.

    Parameters
    ----------
    boxes : list of [x1, y1, x2, y2]
    y_tol : maximum y-centre distance to be considered the same row;
            roughly half a box height is a good value.

    Returns
    -------
    list of rows, each row being a list of [x1,y1,x2,y2] sorted by x1.
    """
    if not boxes:
        return []

    def _yc(b: Box) -> float:
        return (b[1] + b[3]) / 2.0

    sorted_by_y = sorted(boxes, key=_yc)
    rows: list[list[Box]] = []
    current_row: list[Box] = [sorted_by_y[0]]
    row_y_mean: float = _yc(sorted_by_y[0])

    for box in sorted_by_y[1:]:
        cy = _yc(box)
        if abs(cy - row_y_mean) <= y_tol:
            current_row.append(box)
            # Update running mean
            row_y_mean = sum(_yc(b) for b in current_row) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            row_y_mean = cy

    rows.append(sorted(current_row, key=lambda b: b[0]))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-2 – Field assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_fields(rows: list[Row]) -> dict[str, list[Box]]:
    """
    Map the detected rows to named form fields.

    Strategy
    --------
    1. Mobile row is uniquely identified by count (16-24 boxes, no overlap).
    2. Among the remaining rows (with 3-12 boxes each), we use Y-POSITION:
         - topmost row  -> date   (capped to 8 boxes)
         - bottommost row -> cutoff (capped to 5 boxes)
       This avoids any count-range ambiguity when the cutoff area has 6-9
       boxes from decimal / overlapping detections on real scans.
    3. Each row is capped to its target count by keeping the leftmost N
       boxes (left-to-right digit order).
    4. The 20-box mobile row is split at the largest horizontal gap into
       student (10 boxes) and parent (10 boxes).

    Raises
    ------
    ValueError  if the expected rows cannot be identified.
    """
    def _yc_row(r: Row) -> float:
        return sum((b[1] + b[3]) / 2 for b in r) / len(r)

    # ── Step 1: find mobile row by count (16-24), uniquely
    mobile_lo, mobile_hi, mobile_tgt = _ROW_RANGES["mobile"]
    mobile_rows = [r for r in rows if mobile_lo <= len(r) <= mobile_hi]
    if len(mobile_rows) != 1:
        counts = [len(r) for r in sorted(rows, key=_yc_row)]
        raise ValueError(
            f"Expected exactly 1 mobile row (16-24 boxes); "
            f"found {len(mobile_rows)}. All row counts: {counts}"
        )
    mobile_row_raw = mobile_rows[0]

    # ── Step 2: from remaining rows pick date (topmost) and cutoff (bottommost)
    others = [r for r in rows if r is not mobile_row_raw and 3 <= len(r) <= 12]
    if len(others) < 2:
        counts = [len(r) for r in sorted(rows, key=_yc_row)]
        raise ValueError(
            f"Expected at least 2 non-mobile rows (date + cutoff); "
            f"found {len(others)} with 3-12 boxes. All row counts: {counts}"
        )
    others_sorted = sorted(others, key=_yc_row)
    date_row_raw   = others_sorted[0]   # topmost
    cutoff_row_raw = others_sorted[-1]  # bottommost

    if date_row_raw is cutoff_row_raw:
        raise ValueError("date and cutoff resolved to the same row.")

    # ── Step 3: cap to target counts (leftmost N boxes)
    def _cap(row: Row, field: str) -> Row:
        _, _, target = _ROW_RANGES[field]
        return sorted(row, key=lambda b: b[0])[:target]

    date_row   = _cap(date_row_raw,   "date")
    mobile_row = _cap(mobile_row_raw, "mobile")
    cutoff_row = _cap(cutoff_row_raw, "cutoff")

    # ── Split 20-box row at the largest horizontal gap ──
    # Gap = left edge of box[i+1] minus right edge of box[i]
    gaps = [mobile_row[i + 1][0] - mobile_row[i][2]
            for i in range(len(mobile_row) - 1)]

    # Primary: use the position of the maximum gap.
    max_gap_idx  = int(np.argmax(gaps))
    max_gap_val  = gaps[max_gap_idx]

    # Safety fallback: the form is always 10 student + 10 parent boxes.
    # If the expected split point (index 9, i.e. after box 9) has a gap
    # that is at least 60 % of the largest gap, prefer the structural split.
    EXPECTED_SPLIT = 10   # boxes 0-9 = student, 10-19 = parent
    gap_at_expected = gaps[EXPECTED_SPLIT - 1]   # gap between box 9 and 10

    if gap_at_expected >= 0.60 * max_gap_val:
        split_idx = EXPECTED_SPLIT
    else:
        split_idx = max_gap_idx + 1   # first index of parent-mobile

    student_boxes = mobile_row[:split_idx]
    parent_boxes  = mobile_row[split_idx:]

    # Sanity: we expect exactly 10 + 10
    if len(student_boxes) != 10 or len(parent_boxes) != 10:
        raise ValueError(
            f"Mobile row did not split 10/10 (split at {split_idx}): got "
            f"{len(student_boxes)}/{len(parent_boxes)}."
        )

    return {
        "date":           date_row,
        "student_mobile": student_boxes,
        "parent_mobile":  parent_boxes,
        "cutoff":         cutoff_row,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-3 – Box reading
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_box(
    img: np.ndarray,
    box: Box,
    ink_thr: int | None = None,
    box_idx: int | None = None,
) -> np.ndarray:
    """
    Crop a single box from *img*, apply preprocessing, return 28x28 uint8
    (white digit on black background) ready for predict_digit.

    All parameters (threshold, insets, CC filtering, resize) are driven by
    digit_pipeline/preprocess_config.json (written by calibrate_preprocessing.py).
    Falls back to safe built-in defaults when the config file is absent.

    Parameters
    ----------
    img      : full form image (BGR or grayscale)
    box      : [x1, y1, x2, y2] in img coordinates
    ink_thr  : per-form ink threshold (from _compute_form_ink_threshold);
               if None the config global fallback is used.
    box_idx  : 0-based index in the 33-box sequence; used for per-box
               extra left inset (e.g. date[0], cutoff[0]).
    """
    cfg = _get_preprocess_config()
    insets_cfg = cfg.get("edge_insets", {"top": 2, "bottom": 2, "left": 2, "right": 2})
    extra_left_cfg = cfg.get("extra_left_inset", {})
    illum  = cfg.get("illumination_method", "raw_multiotsu")
    tgt    = int(cfg.get("target_size", 28))
    inner  = int(cfg.get("inner_size",  20))
    do_com = bool(cfg.get("center_of_mass_shift", True))
    min_cc_area = int(cfg.get("min_component_area", 5))
    max_edge_fr = float(cfg.get("max_edge_frac_for_line", 0.5))
    min_asp_line= float(cfg.get("min_aspect_ratio_for_line", 3.0))

    # Per-box extra left inset (e.g. '0' -> date[0], '28' -> cutoff[0])
    extra_left = 0
    if box_idx is not None:
        extra_left = int(extra_left_cfg.get(str(box_idx), 0))

    x1, y1, x2, y2 = box
    it = int(insets_cfg.get("top",    2))
    ib = int(insets_cfg.get("bottom", 2))
    il = int(insets_cfg.get("left",   2)) + extra_left
    ir = int(insets_cfg.get("right",  2))

    px1 = max(0, x1 + il); py1 = max(0, y1 + it)
    px2 = min(img.shape[1], x2 - ir)
    py2 = min(img.shape[0], y2 - ib)

    if px2 <= px1 or py2 <= py1:
        return np.zeros((tgt, tgt), dtype=np.uint8)

    crop = img[py1:py2, px1:px2]

    # Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Illumination normalisation
    if illum == "flatten_multiotsu":
        ksize = int(cfg.get("illumination_flatten_ksize", 51)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        bg = np.clip(bg, 1, 255)
        gray = (gray.astype(np.float32) / bg * 255).clip(0, 255).astype(np.uint8)
    elif illum == "adaptive":
        block = int(cfg.get("adaptive_block_size", 51)) | 1
        C_val = int(cfg.get("adaptive_C", 10))
        return cv2.resize(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, C_val),
            (tgt, tgt), interpolation=cv2.INTER_AREA,
        )

    # Threshold (ink dark -> white; background+ghost -> black)
    thr = ink_thr if ink_thr is not None else int(cfg.get("ink_threshold_global", 100))
    _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    # Connected-component filtering
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_bin, w_bin = binary.shape
    keep_ratio = float(cfg.get("component_keep_ratio", 0.15))
    candidates = []
    for i in range(1, n_cc):
        cx, cy, cw, ch, area = stats[i, :5]
        on_edge = int(cx <= 0) + int(cy <= 0) + int(cx + cw >= w_bin) + int(cy + ch >= h_bin)
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        is_line = on_edge >= 2 and aspect >= min_asp_line and (on_edge / 4.0) >= max_edge_fr
        if is_line or area < min_cc_area:
            binary[labels == i] = 0
        else:
            candidates.append((i, int(area)))
    if candidates:
        largest = max(a for _, a in candidates)
        for i, a in candidates:
            if a < keep_ratio * largest:  # drop small fragments
                binary[labels == i] = 0

    # Resize to target_size x target_size
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((tgt, tgt), dtype=np.uint8)

    # Tight-crop ink bounding box
    ink_crop = binary[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ih, iw   = ink_crop.shape
    scale    = inner / max(ih, iw, 1)
    new_h    = max(1, int(round(ih * scale)))
    new_w    = max(1, int(round(iw * scale)))
    ink_r    = cv2.resize(ink_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tgt, tgt), dtype=np.uint8)
    yo = (tgt - new_h) // 2; xo = (tgt - new_w) // 2
    canvas[yo:yo+new_h, xo:xo+new_w] = ink_r

    # Centre-of-mass shift (capped at +-3 px)
    if do_com:
        m = cv2.moments(canvas)
        if m["m00"] > 0:
            cx_m = m["m10"] / m["m00"]; cy_m = m["m01"] / m["m00"]
            dx = int(round(tgt/2 - cx_m)); dy = int(round(tgt/2 - cy_m))
            dx = max(-3, min(3, dx));      dy = max(-3, min(3, dy))
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, M_shift, (tgt, tgt))

    return canvas


def read_field(
    img: np.ndarray,
    boxes: list[Box],
    ink_thr: int | None = None,
    box_idx_start: int = 0,
) -> list[Digit]:
    """
    For each box in *boxes* (in order, left-to-right):
      1. Preprocess to 28x28 binary image using config-driven pipeline.
      2. Call predict_digit.
      3. Return the list of (digit_str, confidence) pairs.

    Parameters
    ----------
    img           : full form image
    boxes         : list of [x1,y1,x2,y2]
    ink_thr       : per-form ink threshold (from _compute_form_ink_threshold)
    box_idx_start : global index of boxes[0] in the 33-box sequence (for
                    per-box extra-left-inset lookup)
    """
    return [
        predict_digit(_preprocess_box(img, box, ink_thr=ink_thr,
                                      box_idx=box_idx_start + i))
        for i, box in enumerate(boxes)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-4 – Build formatted results
# ══════════════════════════════════════════════════════════════════════════════

def build_results(
    fields: dict[str, list[Box]],
    field_digits: dict[str, list[Digit]],
) -> dict[str, str]:
    """
    Join raw digit/confidence pairs into formatted field strings.

    date          -> "DD/MM/YYYY"
    student_mobile -> "XXXXXXXXXX" (10 chars; blank -> "?")
    parent_mobile  -> "XXXXXXXXXX"
    cutoff         -> "ABC.DE"
                      (blank first digit treated as "0";
                       decimal point inserted after position 3)
    """
    def _join(digits: list[Digit], blank_char: str = "?") -> str:
        return "".join(d if d else blank_char for d, _ in digits)

    # ── Date: 8 digits -> DD / MM / YYYY
    date_raw = _join(field_digits["date"])
    date_str = f"{date_raw[0:2]}/{date_raw[2:4]}/{date_raw[4:8]}"

    # ── Mobile numbers: 10 digits each
    student_mob = _join(field_digits["student_mobile"])
    parent_mob  = _join(field_digits["parent_mobile"])

    # ── Cutoff: 5 boxes -> ABC.DE
    co_digits = field_digits["cutoff"]
    co_chars: list[str] = []
    for i, (d, _) in enumerate(co_digits):
        if i == 0 and d == "":
            co_chars.append("0")    # blank first digit -> leading zero
        else:
            co_chars.append(d if d else "?")
    cutoff_str = "".join(co_chars[:3]) + "." + "".join(co_chars[3:5])

    return {
        "date":           date_str,
        "student_mobile": student_mob,
        "parent_mobile":  parent_mob,
        "cutoff":         cutoff_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-5 – Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate(
    result: dict[str, str],
    field_digits: dict[str, list[Digit]],
) -> tuple[list[str], bool]:
    """
    Check the extracted values for logical correctness.

    Checks
    ------
    * date         – real calendar date (DDMMYYYY must parse)
    * student/parent mobile – match [6-9]\d{9}
    * cutoff       – numeric value in [0, 200]
    * confidence   – any digit below CONFIDENCE_THRESHOLD flagged

    Returns
    -------
    issues       : list of human-readable issue strings
    needs_review : True if any hard error or low-confidence digit found
    """
    issues: list[str] = []
    needs_review = False

    # ── Date
    date_str = result.get("date", "")
    clean_date = date_str.replace("/", "")
    if "?" in clean_date or len(clean_date) != 8:
        issues.append(f"Date incomplete or unreadable: '{date_str}'")
        needs_review = True
    else:
        try:
            datetime.strptime(clean_date, "%d%m%Y")
        except ValueError:
            issues.append(f"Date is not a valid calendar date: '{date_str}'")
            needs_review = True

    # ── Mobile numbers
    for field_name in ("student_mobile", "parent_mobile"):
        mob = result.get(field_name, "")
        if not re.fullmatch(r"[6-9]\d{9}", mob):
            issues.append(
                f"'{field_name}' does not look like a valid Indian mobile: '{mob}'"
            )
            needs_review = True

    # ── Cutoff
    cutoff_str = result.get("cutoff", "")
    try:
        cutoff_val = float(cutoff_str.replace("?", "0"))
        if not (0.0 <= cutoff_val <= 200.0):
            issues.append(f"Cutoff out of range [0, 200]: '{cutoff_str}'")
            needs_review = True
    except ValueError:
        issues.append(f"Cutoff is not numeric: '{cutoff_str}'")
        needs_review = True

    # ── Low-confidence digits
    for field_name, digits in field_digits.items():
        for idx, (digit, conf) in enumerate(digits):
            if digit != "" and conf < CONFIDENCE_THRESHOLD:
                issues.append(
                    f"Low confidence on {field_name}[{idx}]: "
                    f"digit='{digit}', conf={conf:.2f}"
                )
                needs_review = True

    return issues, needs_review


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-6 – Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_form(
    path: str,
    model_weights: str = "runs/box_detector/v1/weights/best.pt",
    y_tol: int | None = None,
    save_json: bool = True,
) -> dict[str, Any]:
    """
    End-to-end processing of a single form scan.

    Parameters
    ----------
    path           : path to the scan image (JPEG / PNG)
    model_weights  : path to YOLO weights file
    y_tol          : row-grouping tolerance in px (auto-estimated if None)
    save_json      : if True, writes <stem>_result.json alongside the image

    Returns
    -------
    {
      "data":         { "date": ..., "student_mobile": ...,
                        "parent_mobile": ..., "cutoff": ... },
      "issues":       [ "..." ],
      "needs_review": bool,
    }
    """
    from ultralytics import YOLO   # imported here to keep module importable
                                   # without ultralytics installed (for unit tests)

    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Scan not found: {path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"OpenCV could not read: {path}")

    # ── Step 1: YOLO detection
    model   = YOLO(model_weights)
    results = model.predict(
        source=str(img_path),
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )
    raw_boxes: list[Box] = []
    for r in results:
        for xyxy in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            raw_boxes.append([x1, y1, x2, y2])

    if not raw_boxes:
        return {
            "data":         {},
            "issues":       ["YOLO detected zero boxes."],
            "needs_review": True,
        }

    # ── Step 2: Row grouping
    # Auto-estimate y_tol as half the median box height
    if y_tol is None:
        heights  = [(b[3] - b[1]) for b in raw_boxes]
        y_tol    = max(5, int(np.median(heights) * 0.5))

    rows = to_rows(raw_boxes, y_tol)

    # ── Step 3: Field assignment (Phase 1: homography cutoff projection)
    # Identify date (8) and mobile (20) rows first; use them as homography
    # anchors to project the 5 cutoff cell positions from the template.
    try:
        fields_raw = assign_fields(rows)   # still used for date + mobile rows
    except ValueError as exc:
        return {
            "data":         {},
            "issues":       [str(exc)],
            "needs_review": True,
        }

    date_row_det   = fields_raw["date"]           # up to 8 boxes
    smob_det       = fields_raw["student_mobile"]  # 10 boxes
    pmob_det       = fields_raw["parent_mobile"]   # 10 boxes
    mobile_row_det = smob_det + pmob_det           # 20 boxes

    homography_ok  = (len(date_row_det) == 8 and len(mobile_row_det) == 20)
    needs_review_h = False
    inlier_ratio   = 1.0

    if homography_ok:
        # Collect all detected boxes in the cutoff y-band
        # (same y-range as the template cutoff row, approximated from row data)
        cutoff_row_raw = fields_raw["cutoff"]   # up to 5 boxes (leftmost-5 fallback)
        if cutoff_row_raw:
            cutoff_yc = sum((b[1] + b[3]) / 2 for b in cutoff_row_raw) / len(cutoff_row_raw)
        else:
            cutoff_yc = max((b[1] + b[3]) / 2 for b in raw_boxes)
        # Gather ALL detected boxes within y_tol*3 of the cutoff row centre
        detected_cutoff_all = [
            b for b in raw_boxes
            if abs((b[1] + b[3]) / 2 - cutoff_yc) <= y_tol * 3
        ]

        tpl_boxes = _get_template_boxes()
        proj_cutoff, needs_review_h, inlier_ratio = project_cutoff_via_homography(
            date_row_det, mobile_row_det, tpl_boxes, detected_cutoff_all
        )
        if proj_cutoff and len(proj_cutoff) == 5:
            fields = {
                "date":           date_row_det,
                "student_mobile": smob_det,
                "parent_mobile":  pmob_det,
                "cutoff":         proj_cutoff,
            }
        else:
            # Homography returned nothing usable; fall back
            needs_review_h = True
            fields = fields_raw
    else:
        # Anchor count wrong; fall back and flag
        needs_review_h = True
        fields = fields_raw

    # ── Step 4: Compute per-form ink threshold + Read each field
    # Compute once across ALL 33 boxes so the threshold reflects the full
    # intensity distribution of this particular form scan.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_boxes_33 = (fields["date"] + fields["student_mobile"]
                    + fields["parent_mobile"] + fields["cutoff"])
    ink_thr = _compute_form_ink_threshold(img_gray, all_boxes_33)

    # Field index offsets for per-box extra-left-inset lookup
    _FIELD_BOX_STARTS = {
        "date":           0,
        "student_mobile": 8,
        "parent_mobile":  18,
        "cutoff":         28,
    }
    field_digits: dict[str, list[Digit]] = {
        name: read_field(img, boxes, ink_thr=ink_thr,
                         box_idx_start=_FIELD_BOX_STARTS[name])
        for name, boxes in fields.items()
    }

    # ── Step 5: Build formatted strings
    result = build_results(fields, field_digits)

    # ── Step 6: Validate
    issues, needs_review = validate(result, field_digits)

    # Propagate homography issues
    if needs_review_h:
        issues.append(
            f"Cutoff row projected by homography fallback "
            f"(anchors_ok={homography_ok}, inlier_ratio={inlier_ratio:.2f})."
        )
        needs_review = True

    output = {
        "data":         result,
        "issues":       issues,
        "needs_review": needs_review,
        "_inlier_ratio": round(inlier_ratio, 3),
    }

    # ── Optional JSON dump
    if save_json:
        out_path = img_path.with_name(img_path.stem + "_result.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[process_form] Result -> {out_path}")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-7 – Debug visualisation
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette per field (BGR)
_FIELD_COLORS: dict[str, tuple[int, int, int]] = {
    "date":           (0,   200,  0),    # green
    "student_mobile": (0,   140, 255),   # orange
    "parent_mobile":  (255,  60,  60),   # blue
    "cutoff":         (200,   0, 200),   # magenta
}


def debug_draw(
    img: np.ndarray,
    fields: dict[str, list[Box]],
    field_digits: dict[str, list[Digit]] | None = None,
    out_path: str = "debug_fields.jpg",
) -> None:
    """
    Draw each field's boxes in a distinct colour on a copy of *img*.
    Optionally overlay the predicted digit and confidence.
    Saves the annotated image to *out_path*.
    """
    vis = img.copy()

    for field_name, boxes in fields.items():
        color = _FIELD_COLORS.get(field_name, (200, 200, 200))
        digits_list = (field_digits or {}).get(field_name, [])

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label with field name on the first box of each field
            if idx == 0:
                cv2.putText(vis, field_name, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Overlay digit + confidence if available
            if idx < len(digits_list):
                digit, conf = digits_list[idx]
                txt = digit if digit else "_"
                cv2.putText(vis, f"{txt} {conf:.2f}",
                            (x1 + 2, y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"[debug_draw] Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2-8 – Batch processing
# ══════════════════════════════════════════════════════════════════════════════

def batch_process(
    folder: str,
    out_csv: str = "batch_results.csv",
    model_weights: str = "runs/box_detector/v1/weights/best.pt",
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".bmp"),
) -> None:
    """
    Process every image in *folder*, write one CSV with columns:
        filename, date, student_mobile, parent_mobile, cutoff,
        needs_review, issues
    """
    folder_path = Path(folder)
    images = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        print(f"[batch_process] No images found in '{folder}'.")
        return

    fieldnames = [
        "filename", "date", "student_mobile",
        "parent_mobile", "cutoff", "needs_review", "issues",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in images:
            print(f"[batch_process] Processing: {img_path.name}")
            try:
                result = process_form(
                    str(img_path),
                    model_weights=model_weights,
                    save_json=True,
                )
            except Exception as exc:
                result = {
                    "data":         {},
                    "issues":       [str(exc)],
                    "needs_review": True,
                }

            data = result.get("data", {})
            writer.writerow({
                "filename":       img_path.name,
                "date":           data.get("date", ""),
                "student_mobile": data.get("student_mobile", ""),
                "parent_mobile":  data.get("parent_mobile", ""),
                "cutoff":         data.get("cutoff", ""),
                "needs_review":   result.get("needs_review", True),
                "issues":         "; ".join(result.get("issues", [])),
            })

    print(f"[batch_process] CSV saved -> {out_csv}  ({len(images)} forms)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Process a form scan or a folder of scans."
    )
    sub = parser.add_subparsers(dest="cmd")

    sp = sub.add_parser("single", help="Process one image.")
    sp.add_argument("image",  help="Path to scan image.")
    sp.add_argument("--weights", default="runs/box_detector/v1/weights/best.pt")
    sp.add_argument("--debug", action="store_true",
                    help="Save debug_fields.jpg")

    bp = sub.add_parser("batch", help="Process a folder of scans.")
    bp.add_argument("folder", help="Folder containing scan images.")
    bp.add_argument("--weights", default="runs/box_detector/v1/weights/best.pt")
    bp.add_argument("--out",     default="batch_results.csv")

    args = parser.parse_args()

    if args.cmd == "single":
        # ── Import YOLO lazily so the module works without ultralytics
        output = process_form(args.image, model_weights=args.weights)
        print(json.dumps(output, indent=2))

        if args.debug:
            from ultralytics import YOLO
            img = cv2.imread(args.image)
            model = YOLO(args.weights)
            res = model.predict(
                source=args.image, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False
            )
            raw_boxes: list[Box] = []
            for r in res:
                for xyxy in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, xyxy)
                    raw_boxes.append([x1, y1, x2, y2])

            heights = [(b[3] - b[1]) for b in raw_boxes]
            y_tol   = max(5, int(np.median(heights) * 0.5))
            rows    = to_rows(raw_boxes, y_tol)
            fields  = assign_fields(rows)
            field_digits = {n: read_field(img, b) for n, b in fields.items()}
            debug_draw(img, fields, field_digits)

    elif args.cmd == "batch":
        batch_process(args.folder, out_csv=args.out, model_weights=args.weights)

    else:
        parser.print_help()
        sys.exit(1)
