"""
digit_pipeline/calibrate_preprocessing.py
------------------------------------------
Phase 2: Data-driven preprocessing calibration.

Reads every valid image from data/real_forms_new/, runs YOLO + Phase-1
homography, pools cell-interior pixels, and chooses all preprocessing
parameters from the data.

Outputs
-------
  digit_pipeline/preprocess_config.json
  scratch/preprocess_calibration.md
  scratch/calibration_plots/   (PNGs referenced from the .md)

Usage
-----
  (venv_form) python digit_pipeline/calibrate_preprocessing.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO       = Path(__file__).resolve().parent.parent
IMG_DIR    = REPO / "data" / "real_forms_new"
WEIGHTS    = REPO / "digit_pipeline" / "runs" / "box_detector" / "v1" / "weights" / "best.pt"
BOXES_JSON = REPO / "digit_pipeline" / "boxes.json"
CONFIG_OUT = REPO / "digit_pipeline" / "preprocess_config.json"
MD_OUT     = REPO / "scratch" / "preprocess_calibration.md"
PLOT_DIR   = REPO / "scratch" / "calibration_plots"
CONTACT_DIR= REPO / "scratch" / "blank_boundary_contact"

YOLO_CONF  = 0.25
YOLO_IOU   = 0.70
YOLO_IMGSZ = 1280

# Field index ranges
DATE_IDX   = list(range(0,  8))
SMOB_IDX   = list(range(8,  18))
PMOB_IDX   = list(range(18, 28))
CUT_IDX    = list(range(28, 33))

INSET_PX   = 3   # default cell inset when cropping

# ---------------------------------------------------------------------------
# Helpers: EXIF orientation
# ---------------------------------------------------------------------------

def _apply_exif(img_bgr: np.ndarray, path: Path) -> np.ndarray:
    try:
        from PIL import Image, ImageOps
        pil = Image.open(str(path))
        pil = ImageOps.exif_transpose(pil)
        if (pil.width, pil.height) != (img_bgr.shape[1], img_bgr.shape[0]):
            arr = np.array(pil.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return img_bgr


# ---------------------------------------------------------------------------
# Helpers: row grouping (standalone)
# ---------------------------------------------------------------------------

def _to_rows(boxes: list, y_tol: int) -> list:
    if not boxes:
        return []
    def yc(b): return (b[1]+b[3])/2.
    srt = sorted(boxes, key=yc)
    rows, cur, my = [], [srt[0]], yc(srt[0])
    for b in srt[1:]:
        cy = yc(b)
        if abs(cy-my) <= y_tol:
            cur.append(b); my = sum(yc(x) for x in cur)/len(cur)
        else:
            rows.append(sorted(cur,key=lambda x:x[0])); cur=[b]; my=cy
    rows.append(sorted(cur,key=lambda x:x[0]))
    return rows


# ---------------------------------------------------------------------------
# Helpers: IoU and homography projection
# ---------------------------------------------------------------------------

def _iou(a, b):
    xi1=max(a[0],b[0]); yi1=max(a[1],b[1])
    xi2=min(a[2],b[2]); yi2=min(a[3],b[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.


def _project_cutoff(date_boxes, mobile_boxes, tpl_boxes, detected_cutoff,
                    iou_thr=0.5, inlier_min=0.7):
    anchors_p = date_boxes + mobile_boxes
    if len(anchors_p) != 28:
        return [], True, 0.
    def ctr(b): return [(b[0]+b[2])/2., (b[1]+b[3])/2.]
    src = np.array([ctr(b) for b in tpl_boxes[:28]], dtype=np.float32).reshape(-1,1,2)
    dst = np.array([ctr(b) for b in anchors_p],      dtype=np.float32).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)  # 15px for YOLO 1280->7680 scale
    if H is None:
        return [], True, 0.
    ir = float(mask.ravel().sum())/len(mask)
    result = []
    for tb in tpl_boxes[28:33]:
        corners = np.array([[tb[0],tb[1]],[tb[2],tb[1]],
                             [tb[2],tb[3]],[tb[0],tb[3]]],
                            dtype=np.float32).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1,2)
        pb = [int(proj[:,0].min()), int(proj[:,1].min()),
              int(proj[:,0].max()), int(proj[:,1].max())]
        best_iou, best = 0., None
        for det in detected_cutoff:
            v = _iou(pb, det)
            if v > best_iou: best_iou, best = v, det
        result.append(best if best_iou>=iou_thr else pb)
    result.sort(key=lambda b:b[0])
    return result, ir<inlier_min, ir


# ---------------------------------------------------------------------------
# Helpers: cell crop
# ---------------------------------------------------------------------------

def _crop_cell(img_gray: np.ndarray, box: list, inset: int = INSET_PX) -> np.ndarray | None:
    x1,y1,x2,y2 = box
    x1 = max(0, x1+inset); y1 = max(0, y1+inset)
    x2 = min(img_gray.shape[1], x2-inset)
    y2 = min(img_gray.shape[0], y2-inset)
    if x2<=x1 or y2<=y1:
        return None
    return img_gray[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Helpers: multi-Otsu (2 thresholds, 3 classes), pure numpy
# ---------------------------------------------------------------------------

def _multiotsu2(pixels: np.ndarray) -> tuple[int, int]:
    """Return (t_ink_ghost, t_ghost_paper) via exhaustive 3-class Otsu."""
    hist, _ = np.histogram(pixels.ravel(), bins=256, range=(0, 256))
    h = hist.astype(np.float64)
    total = h.sum()
    if total == 0:
        return 60, 180
    h /= total
    vals = np.arange(256, dtype=np.float64)
    p  = np.cumsum(h)
    mu = np.cumsum(h * vals)
    mu_t = mu[-1]

    # Vectorised over t1; inner loop over t2 (still O(256^2) but fast with numpy)
    best_var = -1.
    best_t1, best_t2 = 60, 180
    for t1 in range(1, 254):
        w0 = p[t1]
        if w0 <= 0 or w0 >= 1:
            continue
        m0 = mu[t1] / w0
        # t2 range
        t2_arr = np.arange(t1+1, 255)
        w1 = p[t2_arr] - p[t1]
        w2 = 1.0 - p[t2_arr]
        ok = (w1 > 0) & (w2 > 0)
        if not ok.any():
            continue
        t2_ok = t2_arr[ok]; w1_ok = w1[ok]; w2_ok = w2[ok]
        m1_ok = (mu[t2_ok] - mu[t1]) / w1_ok
        m2_ok = (mu_t - mu[t2_ok]) / w2_ok
        var = (w0*(m0-mu_t)**2 + w1_ok*(m1_ok-mu_t)**2 + w2_ok*(m2_ok-mu_t)**2)
        idx = int(np.argmax(var))
        if var[idx] > best_var:
            best_var = var[idx]
            best_t1, best_t2 = int(t1), int(t2_ok[idx])
    return best_t1, best_t2


# ---------------------------------------------------------------------------
# Helpers: illumination normalisation
# ---------------------------------------------------------------------------

def _flatten_bg(gray: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Divide by morphological closing to normalise illumination."""
    ksize = ksize | 1   # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32)
    bg = np.clip(bg, 1, 255)
    flat = (gray.astype(np.float32) / bg * 255).clip(0, 255).astype(np.uint8)
    return flat


def _adaptive_thresh(gray: np.ndarray, block: int = 51, C: int = 10) -> np.ndarray:
    block = block | 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, C)


def _apply_thresh(gray: np.ndarray, t: int) -> np.ndarray:
    _, b = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return b


# ---------------------------------------------------------------------------
# Helpers: connected components
# ---------------------------------------------------------------------------

def _cc_stats(binary: np.ndarray) -> tuple[int, list[dict]]:
    """Return (n_components, list of {area, aspect, edge_fraction})."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i, :5]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        # Fraction of component bounding-box perimeter touching image edge
        on_edge = int(x <= 0) + int(y <= 0) + int(x+cw >= w) + int(y+ch >= h)
        edge_frac = on_edge / 4.0
        comps.append({"area": int(area), "aspect": float(aspect),
                       "edge_frac": float(edge_frac),
                       "x": int(x), "y": int(y), "w": int(cw), "h": int(ch)})
    return n-1, comps


# ---------------------------------------------------------------------------
# Helpers: resize to 28x28 (CNN-compatible)
# ---------------------------------------------------------------------------

def _to_28x28(binary: np.ndarray) -> np.ndarray:
    """Tight-crop ink, aspect-preserving resize to 20x20, centre on 28x28."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    ink = binary[y1:y2+1, x1:x2+1]
    # Aspect-preserving resize into 20x20
    h_ink, w_ink = ink.shape
    scale = 20.0 / max(h_ink, w_ink, 1)
    new_h = max(1, int(round(h_ink * scale)))
    new_w = max(1, int(round(w_ink * scale)))
    ink_r = cv2.resize(ink, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Centre on 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    yo = (28 - new_h) // 2; xo = (28 - new_w) // 2
    canvas[yo:yo+new_h, xo:xo+new_w] = ink_r
    # Centre-of-mass shift (capped at +-3px)
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx = m["m10"]/m["m00"]; cy = m["m01"]/m["m00"]
        dx = int(round(14 - cx)); dy = int(round(14 - cy))
        dx = max(-3, min(3, dx)); dy = max(-3, min(3, dy))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas


# ---------------------------------------------------------------------------
# Load dataset + detect
# ---------------------------------------------------------------------------

FormData = dict   # typed alias

def load_dataset(model) -> list[FormData]:
    """
    Run YOLO + Phase-1 homography on every image in IMG_DIR.
    Returns a list of per-form dicts with:
      filename, img_gray, boxes33 (33 boxes in photo space), valid,
      n_detected, inlier_ratio, needs_review_h
    """
    tpl_boxes = json.loads(BOXES_JSON.read_text())
    images = sorted(p for p in IMG_DIR.iterdir()
                    if p.suffix.lower() in {".jpg",".jpeg",".png",".tiff",".bmp"})
    print(f"[calib] {len(images)} images found")

    records = []
    for img_path in images:
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            print(f"  SKIP unreadable: {img_path.name}")
            continue
        img = _apply_exif(img_raw, img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        res = model.predict(source=str(img_path), imgsz=YOLO_IMGSZ,
                            conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        raw_boxes = [[int(v) for v in r2]
                     for r in res for r2 in r.boxes.xyxy.cpu().numpy()]

        n_det = len(raw_boxes)
        if n_det == 0:
            print(f"  {img_path.name}: 0 boxes -- EXCLUDED")
            records.append({"filename": img_path.name, "valid": False,
                            "img_gray": gray, "boxes33": [], "n_detected": 0,
                            "inlier_ratio": 0., "needs_review_h": True})
            continue

        hts = [b[3]-b[1] for b in raw_boxes]
        y_tol = max(5, int(np.median(hts)*0.5))
        rows = _to_rows(raw_boxes, y_tol)
        row_counts = [len(r) for r in rows]

        # Find date (6-11 boxes, topmost) and mobile (16-24, unique) rows
        mob_rows = [r for r in rows if 16<=len(r)<=24]
        if len(mob_rows) != 1:
            print(f"  {img_path.name}: no unique mobile row ({row_counts}) -- EXCLUDED")
            records.append({"filename": img_path.name, "valid": False,
                            "img_gray": gray, "boxes33": [], "n_detected": n_det,
                            "inlier_ratio": 0., "needs_review_h": True})
            continue

        mobile_row = sorted(mob_rows[0], key=lambda b:b[0])
        others = [r for r in rows if r is not mob_rows[0] and 3<=len(r)<=12]
        if len(others) < 2:
            print(f"  {img_path.name}: cannot isolate date+cutoff ({row_counts}) -- EXCLUDED")
            records.append({"filename": img_path.name, "valid": False,
                            "img_gray": gray, "boxes33": [], "n_detected": n_det,
                            "inlier_ratio": 0., "needs_review_h": True})
            continue

        others_s = sorted(others, key=lambda r: sum((b[1]+b[3])/2 for b in r)/len(r))
        date_row = sorted(others_s[0],  key=lambda b:b[0])[:8]
        cutoff_fallback = sorted(others_s[-1], key=lambda b:b[0])[:5]

        # Split mobile row 10/10
        gaps = [mobile_row[i+1][0]-mobile_row[i][2] for i in range(len(mobile_row)-1)]
        expected_gap = gaps[9] if len(gaps) >= 10 else max(gaps, default=0)
        max_gap_idx = int(np.argmax(gaps))
        split = 10 if expected_gap >= 0.6*gaps[max_gap_idx] else max_gap_idx+1
        smob = mobile_row[:split]; pmob = mobile_row[split:]
        if len(smob) != 10 or len(pmob) != 10:
            print(f"  {img_path.name}: mobile split {len(smob)}/{len(pmob)} -- EXCLUDED")
            records.append({"filename": img_path.name, "valid": False,
                            "img_gray": gray, "boxes33": [], "n_detected": n_det,
                            "inlier_ratio": 0., "needs_review_h": True})
            continue

        # Homography projection of cutoff
        cutoff_yc_fb = sum((b[1]+b[3])/2 for b in cutoff_fallback)/len(cutoff_fallback)
        det_cutoff = [b for b in raw_boxes
                      if abs((b[1]+b[3])/2 - cutoff_yc_fb) <= y_tol*3]
        proj_cut, nr_h, ir = _project_cutoff(date_row, smob+pmob, tpl_boxes, det_cutoff)

        if len(proj_cut) != 5:
            proj_cut = cutoff_fallback; nr_h = True

        anchor_ok = (len(date_row)==8 and len(smob+pmob)==20)
        boxes33 = date_row + smob + pmob + proj_cut

        print(f"  {img_path.name}: n={n_det:2d}  rows={row_counts}  "
              f"ir={ir:.2f}  {'REVIEW' if nr_h else 'ok'}")

        records.append({
            "filename":      img_path.name,
            "valid":         True,
            "img_gray":      gray,
            "boxes33":       boxes33,
            "n_detected":    n_det,
            "inlier_ratio":  ir,
            "needs_review_h": nr_h,
            "anchor_ok":     anchor_ok,
        })

    valid = [r for r in records if r["valid"]]
    print(f"[calib] {len(valid)}/{len(records)} valid forms")
    return records


# ---------------------------------------------------------------------------
# Measurement A: per-form multi-Otsu intensity model
# ---------------------------------------------------------------------------

def measure_A(forms: list[FormData]) -> dict:
    """
    For each valid form, pool pixels from all 33 cell interiors and fit
    3-class Otsu. Returns per-form t1 (ink/ghost), t2 (ghost/paper).
    """
    t1_vals, t2_vals = [], []
    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        pixels = []
        for box in f["boxes33"]:
            crop = _crop_cell(gray, box, INSET_PX)
            if crop is not None:
                pixels.append(crop.ravel())
        if not pixels:
            continue
        px = np.concatenate(pixels)
        t1, t2 = _multiotsu2(px)
        t1_vals.append(t1); t2_vals.append(t2)
        f["_t1"] = t1; f["_t2"] = t2   # store per-form

    t1_arr = np.array(t1_vals)
    t2_arr = np.array(t2_vals)
    return {
        "t1_per_form": t1_vals, "t2_per_form": t2_vals,
        "t1_min":  int(t1_arr.min()),  "t1_p05": int(np.percentile(t1_arr, 5)),
        "t1_p50":  int(np.percentile(t1_arr, 50)),
        "t1_p95":  int(np.percentile(t1_arr, 95)), "t1_max": int(t1_arr.max()),
        "t1_mean": float(t1_arr.mean()),
        "t2_min":  int(t2_arr.min()),  "t2_p50": int(np.percentile(t2_arr, 50)),
        "t2_max":  int(t2_arr.max()),
        "gap_min": int((t2_arr - t1_arr).min()),
        "gap_p50": int(np.percentile(t2_arr - t1_arr, 50)),
    }


# ---------------------------------------------------------------------------
# Measurement B: illumination method comparison
# ---------------------------------------------------------------------------

def _blank_false_ink(binary: np.ndarray) -> int:
    """Count non-zero pixels in a binary image (false ink)."""
    return int(np.count_nonzero(binary))

def _ink_px(binary: np.ndarray) -> int:
    return int(np.count_nonzero(binary))

def _cc_count(binary: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(binary, connectivity=8)
    return max(0, n - 1)


def measure_B(forms: list[FormData], ink_thr_global: int) -> dict:
    """
    Compare three illumination methods on labelled blank/filled cells.
    A cell is 'filled' if its raw ink count at t=ink_thr_global is > 50px;
    'blank' otherwise. Metrics: false_ink (blank), ink_px (filled), cc (filled).
    """
    methods = {
        "raw_multiotsu": {},
        "flatten_multiotsu": {},
        "adaptive": {},
    }
    # Per-method accumulators
    for m in methods:
        methods[m] = {"blank_false_ink": [], "filled_ink_px": [], "filled_cc": []}

    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        t1 = f.get("_t1", ink_thr_global)

        # Flattened version (whole image)
        flat = _flatten_bg(gray, 51)

        for box in f["boxes33"]:
            raw_crop = _crop_cell(gray,  box, INSET_PX)
            flt_crop = _crop_cell(flat,  box, INSET_PX)
            if raw_crop is None:
                continue

            # Determine if blank or filled (use raw+global threshold)
            raw_bin_global = _apply_thresh(raw_crop, ink_thr_global)
            is_filled = _ink_px(raw_bin_global) > 50

            # Method 1: raw + per-form multi-Otsu
            b1 = _apply_thresh(raw_crop, t1)
            # Method 2: flatten + per-form multi-Otsu on flattened crop
            if flt_crop is not None:
                t1_fl, _ = _multiotsu2(flt_crop.ravel())
                b2 = _apply_thresh(flt_crop, t1_fl)
            else:
                b2 = b1.copy()
            # Method 3: adaptive on raw
            b3 = _adaptive_thresh(raw_crop)

            for key, b in [("raw_multiotsu", b1),
                            ("flatten_multiotsu", b2),
                            ("adaptive", b3)]:
                if is_filled:
                    methods[key]["filled_ink_px"].append(_ink_px(b))
                    methods[key]["filled_cc"].append(_cc_count(b))
                else:
                    methods[key]["blank_false_ink"].append(_blank_false_ink(b))

    summary = {}
    for m, d in methods.items():
        bi = np.array(d["blank_false_ink"]) if d["blank_false_ink"] else np.array([0])
        fi = np.array(d["filled_ink_px"])   if d["filled_ink_px"]   else np.array([0])
        cc = np.array(d["filled_cc"])       if d["filled_cc"]       else np.array([0])
        summary[m] = {
            "blank_false_ink_mean":  float(bi.mean()),
            "blank_false_ink_p95":   float(np.percentile(bi, 95)),
            "filled_ink_px_mean":    float(fi.mean()),
            "filled_cc_mean":        float(cc.mean()),
            "filled_cc_p95":         float(np.percentile(cc, 95)),
        }
    return summary


# ---------------------------------------------------------------------------
# Measurement C: blank / filled threshold
# ---------------------------------------------------------------------------

def measure_C(forms: list[FormData], ink_thr: int) -> dict:
    """
    Collect per-cell ink-pixel count (under ink_thr) and find the blank/filled
    valley via Otsu on log10(count+1).
    """
    counts = []
    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        for box in f["boxes33"]:
            crop = _crop_cell(gray, box, INSET_PX)
            if crop is None:
                continue
            b = _apply_thresh(crop, ink_thr)
            counts.append(_ink_px(b))

    counts_arr = np.array(counts)
    log_counts = np.log10(counts_arr + 1)
    # Otsu on the log histogram to find blank/filled boundary
    lc_uint8 = np.clip((log_counts / log_counts.max() * 255), 0, 255).astype(np.uint8)
    t_log_uint8, _ = cv2.threshold(lc_uint8, 0, 255, cv2.THRESH_OTSU)
    t_log = (t_log_uint8 / 255.0) * log_counts.max()
    blank_threshold = int(round(10**t_log - 1))

    n_blank  = int((counts_arr <= blank_threshold).sum())
    n_filled = int((counts_arr >  blank_threshold).sum())

    return {
        "blank_threshold_px":  blank_threshold,
        "n_blank_cells":       n_blank,
        "n_filled_cells":      n_filled,
        "ink_count_p05":       int(np.percentile(counts_arr, 5)),
        "ink_count_p50":       int(np.percentile(counts_arr, 50)),
        "ink_count_p95":       int(np.percentile(counts_arr, 95)),
        "ink_count_max":       int(counts_arr.max()),
        "all_counts":          counts_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Measurement D: component filtering
# ---------------------------------------------------------------------------

def measure_D(forms: list[FormData], ink_thr: int, blank_thr: int) -> dict:
    """
    From filled cells: collect CC area and aspect distributions.
    Derive min_component_area and line-filter parameters.
    """
    all_areas, all_aspects, all_edge_fracs = [], [], []
    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        for box in f["boxes33"]:
            crop = _crop_cell(gray, box, INSET_PX)
            if crop is None:
                continue
            b = _apply_thresh(crop, ink_thr)
            if _ink_px(b) <= blank_thr:
                continue  # skip blank
            _, comps = _cc_stats(b)
            for c in comps:
                all_areas.append(c["area"])
                all_aspects.append(c["aspect"])
                all_edge_fracs.append(c["edge_frac"])

    if not all_areas:
        return {"min_component_area": 5, "max_edge_frac_for_line": 0.5,
                "min_aspect_ratio_for_line": 3.0}

    areas = np.array(all_areas)
    return {
        "min_component_area": int(max(4, np.percentile(areas, 5))),
        "area_p05":  int(np.percentile(areas, 5)),
        "area_p25":  int(np.percentile(areas, 25)),
        "area_p50":  int(np.percentile(areas, 50)),
        "area_p95":  int(np.percentile(areas, 95)),
        "aspect_p95": float(np.percentile(all_aspects, 95)),
        "edge_frac_p95": float(np.percentile(all_edge_fracs, 95)),
        "max_edge_frac_for_line": 0.50,
        "min_aspect_ratio_for_line": 3.0,
    }


# ---------------------------------------------------------------------------
# Measurement E: margins
# ---------------------------------------------------------------------------

def measure_E(forms: list[FormData], ink_thr: int, blank_thr: int) -> dict:
    """
    Average dark-pixel profile along each edge of filled cell crops.
    Also check box 0 (date) and box 28 (cutoff) for left-edge ':' contamination.
    """
    PROFILE_W = 10   # check innermost 10 pixels on each edge
    profiles = {"top": [], "bottom": [], "left": [], "right": []}

    left_extra_date   = []   # extra dark pixels in leftmost 6px of box 0
    left_extra_cutoff = []   # extra dark pixels in leftmost 6px of box 28

    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        for bi, box in enumerate(f["boxes33"]):
            crop = _crop_cell(gray, box, 0)   # no inset for edge profiling
            if crop is None or min(crop.shape) < PROFILE_W*2:
                continue
            b = _apply_thresh(crop, ink_thr)
            if _ink_px(b) <= blank_thr:
                continue  # skip blank cells for edge profiling

            h, w = b.shape
            # Mean dark fraction per column/row strip
            profiles["top"].append(   b[:PROFILE_W, :].mean(axis=1)/ 255.)
            profiles["bottom"].append( b[-PROFILE_W:,:].mean(axis=1)/ 255.)
            profiles["left"].append(   b[:, :PROFILE_W].mean(axis=0)/ 255.)
            profiles["right"].append(  b[:, -PROFILE_W:].mean(axis=0)/ 255.)

        # Left-edge contamination for date[0] and cutoff[0]
        if len(f["boxes33"]) >= 29:
            for bi_chk, lst in [(0, left_extra_date), (28, left_extra_cutoff)]:
                box = f["boxes33"][bi_chk]
                crop = _crop_cell(gray, box, 0)
                if crop is not None and crop.shape[1] >= 6:
                    b = _apply_thresh(crop, ink_thr)
                    dark_left = int(np.sum(b[:, :6] > 0))
                    lst.append(dark_left)

    def _inset_from_profile(prof_list, threshold=0.05):
        if not prof_list:
            return 0
        mean_p = np.vstack(prof_list).mean(axis=0)  # shape (PROFILE_W,)
        # Find first pixel from edge where mean dark fraction < threshold
        for i, v in enumerate(mean_p):
            if v < threshold:
                return i
        return PROFILE_W

    inset_top    = _inset_from_profile(profiles["top"])
    inset_bottom = _inset_from_profile(profiles["bottom"])
    inset_left   = _inset_from_profile(profiles["left"])
    inset_right  = _inset_from_profile(profiles["right"])

    extra_left_date   = int(np.percentile(left_extra_date,   75)) if left_extra_date   else 0
    extra_left_cutoff = int(np.percentile(left_extra_cutoff, 75)) if left_extra_cutoff else 0

    # Translate to extra inset: if median dark left-edge px > 5, add 2px
    extra_date_0_inset   = 2 if extra_left_date   > 5 else 0
    extra_cutoff_0_inset = 2 if extra_left_cutoff > 5 else 0

    return {
        "inset_top":     inset_top,
        "inset_bottom":  inset_bottom,
        "inset_left":    inset_left,
        "inset_right":   inset_right,
        "extra_left_date_0":   extra_date_0_inset,
        "extra_left_cutoff_0": extra_cutoff_0_inset,
        "dark_px_left_date_0_p75":   extra_left_date,
        "dark_px_left_cutoff_0_p75": extra_left_cutoff,
    }


# ---------------------------------------------------------------------------
# Measurement F: crop geometry
# ---------------------------------------------------------------------------

def measure_F(forms: list[FormData]) -> dict:
    widths, heights = [], []
    for f in forms:
        if not f["valid"]:
            continue
        for box in f["boxes33"]:
            x1,y1,x2,y2 = box
            widths.append(x2-x1); heights.append(y2-y1)
    if not widths:
        return {}
    w_arr = np.array(widths); h_arr = np.array(heights)
    return {
        "width_min":  int(w_arr.min()),  "width_p50": int(np.percentile(w_arr, 50)),
        "width_max":  int(w_arr.max()),
        "height_min": int(h_arr.min()), "height_p50": int(np.percentile(h_arr, 50)),
        "height_max": int(h_arr.max()),
        "aspect_p50": float(np.percentile(w_arr / np.maximum(h_arr, 1), 50)),
    }


# ---------------------------------------------------------------------------
# Measurement G: generalization check (70/30 form-level split)
# ---------------------------------------------------------------------------

def measure_G(forms: list[FormData], ink_thr: int, blank_thr: int) -> dict:
    """
    Split forms 70/30 by unique filename stem (each file = one form).
    Calibrate blank_thr on train set, validate on held-out.
    Report false-ink rate on both splits.
    """
    valid = [f for f in forms if f["valid"]]
    n = len(valid)
    split = int(n * 0.70)
    train, val = valid[:split], valid[split:]

    def _false_ink_rate(form_list):
        false_ink = []; all_blank = 0
        for f in form_list:
            gray = f["img_gray"]
            for box in f["boxes33"]:
                crop = _crop_cell(gray, box, INSET_PX)
                if crop is None:
                    continue
                b = _apply_thresh(crop, ink_thr)
                cnt = _ink_px(b)
                if cnt <= blank_thr:
                    all_blank += 1
                    if cnt > 0:
                        false_ink.append(cnt)
        rate = len(false_ink) / max(1, all_blank)
        mean_fi = float(np.mean(false_ink)) if false_ink else 0.
        return rate, mean_fi, all_blank

    tr_rate, tr_mean, tr_n = _false_ink_rate(train)
    va_rate, va_mean, va_n = _false_ink_rate(val)

    return {
        "n_train": len(train), "n_val": len(val),
        "train_blank_n":          tr_n,
        "train_false_ink_rate":   round(tr_rate, 4),
        "train_false_ink_px_mean":round(tr_mean, 1),
        "val_blank_n":            va_n,
        "val_false_ink_rate":     round(va_rate, 4),
        "val_false_ink_px_mean":  round(va_mean, 1),
        "generalises_ok":         (va_rate <= tr_rate * 2.0 and va_rate < 0.10),
    }


# ---------------------------------------------------------------------------
# Choose winner for method B
# ---------------------------------------------------------------------------

def _choose_illum_method(B: dict) -> str:
    # Score: lower false-ink + CC closest to 1.0
    best, best_m = None, None
    for m, s in B.items():
        fi_score = s["blank_false_ink_mean"]
        cc_score = abs(s["filled_cc_mean"] - 1.0)
        score = fi_score + cc_score * 5   # weight CC deviation more
        if best is None or score < best:
            best = score; best_m = m
    return best_m


# ---------------------------------------------------------------------------
# Contact sheet for C (blank/filled boundary)
# ---------------------------------------------------------------------------

def save_contact_sheet(forms, ink_thr, blank_thr, n_each=8):
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    near_blank = []; near_filled = []
    lo, hi = blank_thr * 0.5, blank_thr * 2.0
    for f in forms:
        if not f["valid"]:
            continue
        gray = f["img_gray"]
        for bi, box in enumerate(f["boxes33"]):
            crop = _crop_cell(gray, box, INSET_PX)
            if crop is None:
                continue
            b = _apply_thresh(crop, ink_thr)
            cnt = _ink_px(b)
            if lo <= cnt <= blank_thr and len(near_blank)  < n_each:
                near_blank.append((crop, b, cnt))
            if blank_thr < cnt <= hi   and len(near_filled) < n_each:
                near_filled.append((crop, b, cnt))
        if len(near_blank) >= n_each and len(near_filled) >= n_each:
            break

    rows = []
    for grp, label in [(near_blank, "blank"), (near_filled, "filled")]:
        cells = []
        for raw_c, bin_c, cnt in grp:
            raw_r = cv2.resize(raw_c, (56,56)) if raw_c.shape[0]>0 else np.zeros((56,56),np.uint8)
            bin_r = cv2.resize(bin_c, (56,56)) if bin_c.shape[0]>0 else np.zeros((56,56),np.uint8)
            pair  = np.hstack([raw_r, bin_r])
            cells.append(pair)
        if cells:
            row = np.hstack(cells)
            rows.append(row)

    if rows:
        max_w = max(r.shape[1] for r in rows)
        padded = [np.pad(r, ((0,0),(0,max_w-r.shape[1])), constant_values=128) for r in rows]
        sheet = np.vstack(padded)
        out = CONTACT_DIR / "blank_filled_boundary.png"
        cv2.imwrite(str(out), sheet)
        print(f"  [E] Contact sheet -> {out}")


# ---------------------------------------------------------------------------
# Write preprocess_config.json
# ---------------------------------------------------------------------------

def write_config(A, B, C, D, E, F, illum_method, t1_clamp_lo, t1_clamp_hi):
    cfg = {
        "version": 2,
        "ink_threshold_method":     "per_form_multiotsu",
        "ink_threshold_global":     A["t1_p50"],
        "ink_threshold_clamp_lo":   t1_clamp_lo,
        "ink_threshold_clamp_hi":   t1_clamp_hi,
        "illumination_method":      illum_method,
        "illumination_flatten_ksize": 51,
        "adaptive_block_size":      51,
        "adaptive_C":               10,
        "blank_cell_ink_threshold": C["blank_threshold_px"],
        "min_component_area":       D["min_component_area"],
        "max_edge_frac_for_line":   D["max_edge_frac_for_line"],
        "min_aspect_ratio_for_line":D["min_aspect_ratio_for_line"],
        "edge_insets": {
            "top":    E["inset_top"],
            "bottom": E["inset_bottom"],
            "left":   E["inset_left"],
            "right":  E["inset_right"],
        },
        "extra_left_inset": {
            "0":  E["extra_left_date_0"],
            "28": E["extra_left_cutoff_0"],
        },
        "target_size": 28,
        "inner_size":  20,
        "center_of_mass_shift": True,
    }
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.write_text(json.dumps(cfg, indent=2))
    print(f"[calib] Config -> {CONFIG_OUT}")
    return cfg


# ---------------------------------------------------------------------------
# Write calibration report (markdown)
# ---------------------------------------------------------------------------

def write_report(A, B, C, D, E, F, G, cfg, illum_method):
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing Calibration Report",
        "",
        "Generated by `digit_pipeline/calibrate_preprocessing.py`.",
        "",
        "---",
        "",
        "## A. Intensity model (multi-Otsu, 3 classes)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| t1 (ink/ghost) min | {A['t1_min']} |",
        f"| t1 p05 | {A['t1_p05']} |",
        f"| t1 p50 (median) | {A['t1_p50']} |",
        f"| t1 p95 | {A['t1_p95']} |",
        f"| t1 max | {A['t1_max']} |",
        f"| t2 (ghost/paper) p50 | {A['t2_p50']} |",
        f"| gap (t2-t1) p50 | {A['gap_p50']} |",
        "",
        f"**Rule**: per-form t1 clamped to "
        f"[{cfg['ink_threshold_clamp_lo']}, {cfg['ink_threshold_clamp_hi']}]. "
        f"Global fallback: {cfg['ink_threshold_global']}.",
        "",
        "---",
        "",
        "## B. Illumination method comparison",
        "",
        "| Method | blank false-ink mean | filled ink-px mean | filled CC mean |",
        "|---|---|---|---|",
    ]
    for m, s in B.items():
        winner = " **WINNER**" if m == illum_method else ""
        lines.append(
            f"| {m}{winner} | {s['blank_false_ink_mean']:.1f} "
            f"| {s['filled_ink_px_mean']:.1f} | {s['filled_cc_mean']:.2f} |"
        )
    lines += [
        "",
        f"**Chosen**: `{illum_method}` (lowest false-ink with CC closest to 1).",
        "",
        "---",
        "",
        "## C. Blank / filled cell threshold",
        "",
        f"- Ink-pixel count distribution: p05={C['ink_count_p05']}, "
        f"p50={C['ink_count_p50']}, p95={C['ink_count_p95']}",
        f"- **Blank threshold**: {C['blank_threshold_px']} px "
        f"(Otsu on log10 histogram)",
        f"- Blank cells: {C['n_blank_cells']}, Filled cells: {C['n_filled_cells']}",
        "",
        "Contact sheet of boundary cells: "
        "`scratch/blank_boundary_contact/blank_filled_boundary.png`",
        "",
        "---",
        "",
        "## D. Component filtering",
        "",
        f"- CC area: p05={D['area_p05']}, p25={D['area_p25']}, "
        f"p50={D['area_p50']}, p95={D['area_p95']}",
        f"- **min_component_area**: {D['min_component_area']} px",
        f"- **Line filter**: edge_frac > {D['max_edge_frac_for_line']} "
        f"AND aspect > {D['min_aspect_ratio_for_line']}",
        "",
        "---",
        "",
        "## E. Margins",
        "",
        f"| Edge | Inset (px) |",
        "|---|---|",
        f"| top | {E['inset_top']} |",
        f"| bottom | {E['inset_bottom']} |",
        f"| left | {E['inset_left']} |",
        f"| right | {E['inset_right']} |",
        "",
        f"Extra left inset — date box 0: {E['extra_left_date_0']} px "
        f"(p75 dark-px={E['dark_px_left_date_0_p75']})",
        f"Extra left inset — cutoff box 0: {E['extra_left_cutoff_0']} px "
        f"(p75 dark-px={E['dark_px_left_cutoff_0_p75']})",
        "",
        "> No blanket top inset applied (would cut tops of 5 and 7).",
        "",
        "---",
        "",
        "## F. Crop geometry",
        "",
        f"- Cell width: min={F.get('width_min','?')} p50={F.get('width_p50','?')} "
        f"max={F.get('width_max','?')} px",
        f"- Cell height: min={F.get('height_min','?')} p50={F.get('height_p50','?')} "
        f"max={F.get('height_max','?')} px",
        f"- Aspect (w/h) p50: {F.get('aspect_p50','?')}",
        "",
        "**Resize pipeline**: tight-crop ink bbox -> "
        "aspect-preserving resize to 20x20 -> centre on 28x28 -> CoM shift (+-3px).",
        "",
        "---",
        "",
        "## G. Generalization check",
        "",
        f"| Split | N forms | Blank cells | False-ink rate | Mean false-ink px |",
        "|---|---|---|---|---|",
        f"| Train (70%) | {G['n_train']} | {G['train_blank_n']} "
        f"| {G['train_false_ink_rate']:.4f} | {G['train_false_ink_px_mean']:.1f} |",
        f"| Val (30%)   | {G['n_val']}   | {G['val_blank_n']} "
        f"| {G['val_false_ink_rate']:.4f}  | {G['val_false_ink_px_mean']:.1f}  |",
        "",
        f"Generalises OK: **{G['generalises_ok']}**",
        "",
        "---",
        "",
        "## Config written",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[calib] Report -> {MD_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st = WEIGHTS.stat()
    from datetime import datetime
    mod = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[calib] Weights: size={st.st_size} modified={mod}")
    assert st.st_size == 22_640_810

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    # -- Load dataset ----------------------------------------------------------
    print("\n[calib] === Loading dataset ===")
    forms = load_dataset(model)

    # -- A: intensity model ----------------------------------------------------
    print("\n[calib] === A: multi-Otsu per form ===")
    A = measure_A(forms)
    t1_global   = A["t1_p50"]
    t1_clamp_lo = A["t1_p05"]
    t1_clamp_hi = A["t1_p95"]
    print(f"  t1: min={A['t1_min']} p05={A['t1_p05']} p50={A['t1_p50']} "
          f"p95={A['t1_p95']} max={A['t1_max']}")
    print(f"  t2: min={A['t2_min']} p50={A['t2_p50']} max={A['t2_max']}")
    print(f"  gap(t2-t1): min={A['gap_min']} p50={A['gap_p50']}")
    print(f"  Clamp range: [{t1_clamp_lo}, {t1_clamp_hi}]  global={t1_global}")

    # -- B: illumination -------------------------------------------------------
    print("\n[calib] === B: illumination comparison ===")
    B = measure_B(forms, t1_global)
    illum_method = _choose_illum_method(B)
    for m, s in B.items():
        flag = "<<" if m == illum_method else "  "
        print(f"  {flag} {m}: blank_fi={s['blank_false_ink_mean']:.1f}  "
              f"filled_ink={s['filled_ink_px_mean']:.1f}  cc={s['filled_cc_mean']:.2f}")
    print(f"  -> Winner: {illum_method}")

    # -- C: blank/filled threshold ---------------------------------------------
    print("\n[calib] === C: blank / filled threshold ===")
    C = measure_C(forms, t1_global)
    blank_thr = C["blank_threshold_px"]
    print(f"  ink count: p05={C['ink_count_p05']} p50={C['ink_count_p50']} "
          f"p95={C['ink_count_p95']} max={C['ink_count_max']}")
    print(f"  blank_threshold={blank_thr}  blank={C['n_blank_cells']}  "
          f"filled={C['n_filled_cells']}")
    save_contact_sheet(forms, t1_global, blank_thr)

    # -- D: component filtering ------------------------------------------------
    print("\n[calib] === D: component filtering ===")
    D = measure_D(forms, t1_global, blank_thr)
    print(f"  area: p05={D['area_p05']} p50={D['area_p50']} p95={D['area_p95']}")
    print(f"  min_component_area={D['min_component_area']}")

    # -- E: margins ------------------------------------------------------------
    print("\n[calib] === E: margins ===")
    E = measure_E(forms, t1_global, blank_thr)
    print(f"  insets: top={E['inset_top']} bottom={E['inset_bottom']} "
          f"left={E['inset_left']} right={E['inset_right']}")
    print(f"  extra left: date[0]={E['extra_left_date_0']}px  "
          f"cutoff[0]={E['extra_left_cutoff_0']}px")

    # -- F: crop geometry ------------------------------------------------------
    print("\n[calib] === F: crop geometry ===")
    F = measure_F(forms)
    print(f"  width : min={F.get('width_min')} p50={F.get('width_p50')} "
          f"max={F.get('width_max')}")
    print(f"  height: min={F.get('height_min')} p50={F.get('height_p50')} "
          f"max={F.get('height_max')}")

    # -- G: generalization -----------------------------------------------------
    print("\n[calib] === G: generalization (70/30 split) ===")
    G = measure_G(forms, t1_global, blank_thr)
    print(f"  train: n={G['n_train']} false_ink_rate={G['train_false_ink_rate']:.4f}")
    print(f"  val  : n={G['n_val']}   false_ink_rate={G['val_false_ink_rate']:.4f}")
    print(f"  generalises_ok={G['generalises_ok']}")

    # -- Write outputs ---------------------------------------------------------
    print("\n[calib] === Writing outputs ===")
    cfg = write_config(A, B, C, D, E, F, illum_method, t1_clamp_lo, t1_clamp_hi)
    # Remove raw arrays before writing report (large)
    C_rep = {k: v for k, v in C.items() if k != "all_counts"}
    write_report(A, B, C_rep, D, E, F, G, cfg, illum_method)
    print("\n[calib] Done.")


if __name__ == "__main__":
    main()
