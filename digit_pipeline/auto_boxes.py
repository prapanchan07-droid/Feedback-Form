"""
auto_boxes.py  (v3 - scan-and-split, calibrated)
─────────────────────────────────────────────────
Detects the 33 digit boxes on the Career Guidance feedback form.

Uses a horizontal-scan approach: at each expected y-band, scan rows
of pixels to find dark x-clusters (the 7-segment '8' inside each box).
Then split each cluster group into N equal sub-boxes.

Expected layout (form_template.pdf rendered at 300 DPI, cropped):
  Date     (8):  y~284-410,  3 groups: DD(144-277), MM(326-459), YYYY(508-771)
  StudentMob(10): y~595-731, 1 group:  x=144-795
  ParentMob (10): y~595-731, 1 group:  x=1260-1913
  Cutoff    (5):  y~856-934, ABC(x~1350-1620) + DE(x~1640-1870) split 3+2

Output:
  digit_pipeline/boxes.json      [[x1,y1,x2,y2], ...]  33 entries
  scratch/boxes_preview.jpg      annotated preview

Usage:
  venv_form\\Scripts\\python.exe digit_pipeline/auto_boxes.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

LABEL_ORDER = [
    "date_d1","date_d2","date_m1","date_m2",
    "date_y1","date_y2","date_y3","date_y4",
    "smob_0","smob_1","smob_2","smob_3","smob_4",
    "smob_5","smob_6","smob_7","smob_8","smob_9",
    "pmob_0","pmob_1","pmob_2","pmob_3","pmob_4",
    "pmob_5","pmob_6","pmob_7","pmob_8","pmob_9",
    "co_a","co_b","co_c","co_d","co_e",
]

# Row groups for size-consistency check
_BOX_GROUPS = [
    ("Date",       list(range(0,  8))),
    ("StudentMob", list(range(8,  18))),
    ("ParentMob",  list(range(18, 28))),
    ("Cutoff",     list(range(28, 33))),
]
SIZE_TOL = 0.15   # 15 % of group median


def validate_box_sizes(boxes: list[list[int]],
                       tol: float = SIZE_TOL) -> bool:
    """
    Check that every box in each row-group is within *tol* (15 %) of the
    group median width and height.  Prints a report and returns False if any
    box fails; returns True when all pass.
    """
    import numpy as np
    ok = True
    for gname, idxs in _BOX_GROUPS:
        ws = [boxes[i][2] - boxes[i][0] for i in idxs]
        hs = [boxes[i][3] - boxes[i][1] for i in idxs]
        mw, mh = float(np.median(ws)), float(np.median(hs))
        for rank, i in enumerate(idxs):
            dw = abs(ws[rank] - mw) / mw
            dh = abs(hs[rank] - mh) / mh
            if dw > tol or dh > tol:
                print(f"[SIZE-CHECK FAIL] {gname} box {i+1} "
                      f"({LABEL_ORDER[i]}): "
                      f"w={ws[rank]} dw={dw:.1%}  h={hs[rank]} dh={dh:.1%}  "
                      f"(tolerance {tol:.0%})")
                ok = False
    if ok:
        print("[SIZE-CHECK] All 33 boxes within "
              f"{tol:.0%} of group medians. OK.")
    return ok


# ─────────────────────────── scan helpers ───────────────────────────

def _find_x_segments(gray: np.ndarray, y_lo: int, y_hi: int,
                     thresh: int = 210, min_width: int = 40
                     ) -> list[tuple[int,int]]:
    """
    Scan rows y_lo..y_hi, collect all dark (< thresh) x-pixels,
    merge into contiguous segments, return those >= min_width as
    (x_start, x_end) tuples.
    """
    h, w = gray.shape
    y_lo = max(0, y_lo);  y_hi = min(h, y_hi)
    acc  = np.zeros(w, dtype=np.int32)
    for y in range(y_lo, y_hi):
        row = (gray[y] < thresh).astype(np.int32)
        acc += row

    # Keep columns dark in at least 2 rows (dotted lines appear in few rows)
    threshold_count = 2
    dark_cols = np.where(acc >= threshold_count)[0]
    if len(dark_cols) == 0:
        return []

    gaps   = np.where(np.diff(dark_cols) > 8)[0]
    splits = np.split(dark_cols, gaps + 1)
    segs   = [(int(s[0]), int(s[-1])) for s in splits if len(s) >= min_width]
    return segs


def _find_y_band(gray: np.ndarray, y_hint: int, half: int = 80,
                 thresh: int = 210, min_dark_cols: int = 50
                 ) -> tuple[int, int]:
    """
    Scan around y_hint ± half to find the actual y-extent of the box row.
    Returns (y_top, y_bot).
    """
    h, w = gray.shape
    y_lo = max(0, y_hint - half)
    y_hi = min(h, y_hint + half)

    # For each y, count dark columns
    counts = []
    for y in range(y_lo, y_hi):
        dark = int((gray[y] < thresh).sum())
        counts.append((y, dark))

    # Find rows with significant darkness
    dark_rows = [y for y, c in counts if c >= min_dark_cols]
    if not dark_rows:
        return (y_hint - 50, y_hint + 50)
    return (min(dark_rows), max(dark_rows))


def _split_segment(x_lo: int, x_hi: int, n: int,
                   y_top: int, y_bot: int) -> list[list[int]]:
    """Split x-segment evenly into n boxes."""
    w     = x_hi - x_lo
    step  = w / n
    boxes = []
    for i in range(n):
        bx1 = int(x_lo + i * step)
        bx2 = int(x_lo + (i + 1) * step)
        boxes.append([bx1, y_top, bx2, y_bot])
    return boxes


# ─────────────────────────── main detector ──────────────────────────

def detect_boxes(template_path: str, out_json: str,
                 preview_path: str) -> bool:
    img  = cv2.imread(template_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read: {template_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape
    print(f"[auto_boxes] Template: {iw}x{ih} px")

    # ── Find reference-strip x-boundary (top-right, exclude) ──────────────
    # The 0-9 strip appears in the top ~15% of the form, right side
    # We exclude x > 60% of width AND y < 15% of height from the scan
    ref_x  = int(iw * 0.67)   # YYYY group ends at ~x=771 / 2480 = 31%, safe
    ref_y  = int(ih * 0.15)

    # ─────────────────────────────────────────────────────── 1. Date row
    # Known from scan: 3 groups of boxes at y~284-410
    date_y_hint = int(ih * 0.21)   # ~21% down from top (347px in 1651)
    date_y_top, date_y_bot = _find_y_band(gray, date_y_hint,
                                           half=80, min_dark_cols=30)
    date_segs = _find_x_segments(gray, date_y_top, date_y_bot,
                                  thresh=210, min_width=40)
    # Filter: must be left of ref_x
    # Date boxes are all on the left half; remove anything > 1000px (ref strip noise)
    date_segs = [(x0,x1) for x0,x1 in date_segs if x0 < 1000]
    # Expect 3 groups (DD, MM, YYYY) with counts 2, 2, 4
    date_group_sizes = [2, 2, 4]
    print(f"[auto_boxes] Date row  y={date_y_top}-{date_y_bot}  "
          f"segments: {date_segs}")

    if len(date_segs) != 3:
        print(f"[FAIL] Expected 3 Date groups, found {len(date_segs)}: {date_segs}")
        _save_fail_preview(img, [], preview_path)
        return False

    date_boxes: list[list[int]] = []
    for (x0, x1), n in zip(date_segs, date_group_sizes):
        date_boxes += _split_segment(x0, x1, n, date_y_top, date_y_bot)

    # ───────────────────────────────────────────── 2. Mobile rows (same y)
    mob_y_hint  = int(ih * 0.40)   # ~40% down (~660px)
    mob_y_top, mob_y_bot = _find_y_band(gray, mob_y_hint,
                                         half=90, min_dark_cols=80)
    mob_segs = _find_x_segments(gray, mob_y_top, mob_y_bot,
                                 thresh=210, min_width=60)
    # Expect exactly 2 wide segments: Student(left) and Parent(right)
    print(f"[auto_boxes] Mobile row y={mob_y_top}-{mob_y_bot}  "
          f"segments: {mob_segs}")

    if len(mob_segs) < 2:
        # Try fallback: split biggest segment in half
        if len(mob_segs) == 1:
            x0, x1 = mob_segs[0]
            mid     = (x0 + x1) // 2
            mob_segs = [(x0, mid), (mid, x1)]
            print(f"  Fallback: split single segment -> {mob_segs}")
        else:
            print(f"[FAIL] Expected 2 mobile groups, found {len(mob_segs)}")
            _save_fail_preview(img, date_boxes, preview_path)
            return False

    # Keep exactly 2 segments (largest two if more)
    if len(mob_segs) > 2:
        mob_segs = sorted(mob_segs, key=lambda s: s[1]-s[0], reverse=True)[:2]
        mob_segs.sort(key=lambda s: s[0])   # left-to-right

    smob_boxes = _split_segment(mob_segs[0][0], mob_segs[0][1], 10,
                                 mob_y_top, mob_y_bot)
    pmob_boxes = _split_segment(mob_segs[1][0], mob_segs[1][1], 10,
                                 mob_y_top, mob_y_bot)

    # ──────────────────────────────────────────────────── 3. Cutoff row
    co_y_hint  = int(ih * 0.55)   # ~55% down (~908px)
    co_y_top, co_y_bot = _find_y_band(gray, co_y_hint,
                                       half=80, min_dark_cols=20)
    co_segs = _find_x_segments(gray, co_y_top, co_y_bot,
                                thresh=210, min_width=30)
    # Filter: cutoff is on the right half of the form
    co_segs_right = [(x0,x1) for x0,x1 in co_segs if x0 > iw * 0.45]
    print(f"[auto_boxes] Cutoff row y={co_y_top}-{co_y_bot}  "
          f"all segs: {co_segs}  right segs: {co_segs_right}")

    # Expect 1 or 2 right-side segments covering 5 boxes total
    # ABC (3 boxes) and DE (2 boxes) are separated by the decimal point
    co_boxes: list[list[int]] = []
    if len(co_segs_right) >= 2:
        # Sort by x, take the two largest
        co_segs_right.sort(key=lambda s: s[1]-s[0], reverse=True)
        seg_abc = co_segs_right[0]
        seg_de  = co_segs_right[1]
        seg_abc, seg_de = sorted([seg_abc, seg_de], key=lambda s: s[0])
        co_boxes  = (_split_segment(seg_abc[0], seg_abc[1], 3,
                                    co_y_top, co_y_bot) +
                     _split_segment(seg_de[0],  seg_de[1],  2,
                                    co_y_top, co_y_bot))
    elif len(co_segs_right) == 1:
        # Single merged group: split 5 evenly (decimal in the middle)
        x0, x1 = co_segs_right[0]
        # Find largest gap in x-range to locate the decimal separator
        acc = np.zeros(iw, dtype=np.int32)
        for y in range(co_y_top, co_y_bot):
            acc += (gray[y, :] < 210).astype(np.int32)
        sub   = acc[x0:x1+1]
        # Find the widest zero-gap inside [x0, x1]
        zero_runs  = []
        in_zero    = False
        start_zero = 0
        for xi, v in enumerate(sub):
            if v == 0 and not in_zero:
                in_zero = True; start_zero = xi
            elif v > 0 and in_zero:
                zero_runs.append((start_zero, xi-1, xi - start_zero))
                in_zero = False
        if in_zero:
            zero_runs.append((start_zero, len(sub)-1, len(sub)-1-start_zero))

        if zero_runs:
            # The decimal point creates the widest gap
            zero_runs.sort(key=lambda r: r[2], reverse=True)
            gap_mid = x0 + (zero_runs[0][0] + zero_runs[0][1]) // 2
            co_boxes = (_split_segment(x0,     gap_mid, 3, co_y_top, co_y_bot) +
                        _split_segment(gap_mid, x1,     2, co_y_top, co_y_bot))
        else:
            co_boxes = _split_segment(x0, x1, 5, co_y_top, co_y_bot)
    else:
        print(f"[FAIL] Could not find Cutoff boxes. co_segs={co_segs}")
        all_so_far = date_boxes + smob_boxes + pmob_boxes
        _save_fail_preview(img, all_so_far, preview_path)
        return False

    # ─────────────────────────────────────── 4. Assemble and validate
    ordered = date_boxes + smob_boxes + pmob_boxes + co_boxes
    if len(ordered) != 33:
        print(f"[FAIL] Final count {len(ordered)} != 33")
        _save_fail_preview(img, ordered, preview_path)
        return False

    # Report
    widths  = [b[2]-b[0] for b in ordered]
    heights = [b[3]-b[1] for b in ordered]
    print(f"\n[auto_boxes] SUCCESS - 33 boxes  "
          f"(w: {min(widths)}-{max(widths)}px  h: {min(heights)}-{max(heights)}px)")
    group_names = (["Date"]*8 + ["StudentMob"]*10 +
                   ["ParentMob"]*10 + ["Cutoff"]*5)
    for i, (x1,y1,x2,y2) in enumerate(ordered):
        print(f"  [{i+1:2d}] {LABEL_ORDER[i]:<12} "
              f"[{x1:4d},{y1:4d},{x2:4d},{y2:4d}]  "
              f"{x2-x1}x{y2-y1}  ({group_names[i]})")

    # Save boxes.json
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    boxes_list = [[x1,y1,x2,y2] for x1,y1,x2,y2 in ordered]
    with open(out_json, "w") as f:
        json.dump(boxes_list, f, indent=2)
    print(f"\n[auto_boxes] boxes.json -> {out_json}")

    # ── Mandatory size-consistency check (15% median tolerance)
    if not validate_box_sizes(boxes_list):
        import sys
        sys.exit("[auto_boxes] ABORT: box sizes failed 15% tolerance. "
                 "Fix detection before proceeding.")

    # Save preview
    _save_ok_preview(img, [date_boxes, smob_boxes, pmob_boxes, co_boxes],
                     preview_path)
    return True



# ─────────────────────────── preview helpers ────────────────────────

def _save_ok_preview(img, row_groups, preview_path):
    colors = [(0,200,0),(200,100,0),(0,100,200),(160,0,160)]
    vis    = img.copy()
    idx    = 0
    for gi, (row, col) in enumerate(zip(row_groups, colors)):
        for b in row:
            x1,y1,x2,y2 = b
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,3)
            cv2.putText(vis, f"{idx+1}:{LABEL_ORDER[idx]}",
                        (x1+2, max(8,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            idx += 1
    bar = np.full((60, vis.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar,
                "AUTO-BOXES OK - 33 boxes  "
                "Green=Date  Blue=StudentMob  Orange=ParentMob  Purple=Cutoff",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
    out = np.vstack([bar, vis])
    _write_preview(out, preview_path)


def _save_fail_preview(img, boxes, preview_path):
    vis = img.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,200),2)
    bar = np.full((60, vis.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, "AUTO-BOXES FAILED - see stdout for details",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 2)
    _write_preview(np.vstack([bar, vis]), preview_path)


def _write_preview(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    max_w = 1400
    if img.shape[1] > max_w:
        sc  = max_w / img.shape[1]
        img = cv2.resize(img,(max_w,int(img.shape[0]*sc)),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"[auto_boxes] Preview -> {path}")


# ────────────────────────────────────────────── CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="digit_pipeline/template_300dpi.png")
    p.add_argument("--out",      default="digit_pipeline/boxes.json")
    p.add_argument("--preview",  default="scratch/boxes_preview.jpg")
    args = p.parse_args()
    ok = detect_boxes(args.template, args.out, args.preview)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
