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

def predict_digit(crop_28x28: np.ndarray) -> tuple[str, float]:
    """
    STUB – returns a random digit (or blank) with random confidence.
    Replace with the real CNN inference call.

    Example real implementation:
        tensor = preprocess(crop_28x28)         # your normalisation
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        digit = str(pred.item()) if conf > 0.35 else ""
        return digit, conf.item()
    """
    if random.random() < 0.15:          # ~15 % blank boxes
        return "", random.uniform(0.20, 0.55)
    digit = str(random.randint(0, 9))
    conf  = random.uniform(0.55, 0.99)
    return digit, conf

# ══════════════════════════════════════════════════════════════════════════════
# Layout constants (matches the form spec)
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_ROW_COUNTS = [8, 20, 5]   # date, mobile-row, cutoff

# Confidence threshold below which a digit is flagged for review
CONFIDENCE_THRESHOLD = 0.80

# YOLO inference settings
YOLO_CONF   = 0.25
YOLO_IMGSZ  = 1280

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

Box   = list[int]          # [x1, y1, x2, y2]
Row   = list[Box]          # sorted left-to-right
Digit = tuple[str, float]  # (digit_str, confidence)


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

    Steps
    -----
    1. Discard stray rows (any row that does NOT have exactly 5, 8, or 20 boxes).
       This eliminates the 0-9 reference strip and any noise detections.
    2. Verify the remaining row counts are exactly [8, 20, 5].
    3. Split the 20-box row at the largest horizontal gap -> student / parent.
    4. Return a dict with keys: date, student_mobile, parent_mobile, cutoff.

    Raises
    ------
    ValueError  if the surviving rows do not match the expected layout.
    """
    valid_counts = set(_EXPECTED_ROW_COUNTS)  # {5, 8, 20}
    kept: list[Row] = [r for r in rows if len(r) in valid_counts]

    # Sort kept rows top-to-bottom by their median y-centre
    kept.sort(key=lambda r: sum((b[1] + b[3]) / 2 for b in r) / len(r))

    # Extract by count
    count_map: dict[int, list[Row]] = {5: [], 8: [], 20: []}
    for row in kept:
        count_map[len(row)].append(row)

    # We expect exactly one row of each count
    if (len(count_map[8])  != 1 or
            len(count_map[20]) != 1 or
            len(count_map[5])  != 1):
        raise ValueError(
            f"Unexpected row layout after filtering. "
            f"Got counts: {[len(r) for r in kept]} "
            f"(expected {_EXPECTED_ROW_COUNTS})."
        )

    date_row   = count_map[8][0]
    mobile_row = count_map[20][0]
    cutoff_row = count_map[5][0]

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

def _preprocess_box(img: np.ndarray, box: Box, pad: int = 3) -> np.ndarray:
    """
    Crop a single box from *img*, apply preprocessing, return 28×28 uint8
    (white digit on black background) ready for predict_digit.

    Steps
    -----
    1. Crop with *pad* px inset on each side.
    2. Grayscale conversion.
    3. Binary inverse threshold at 120 (removes light-grey guide/ghost digits).
    4. Resize to 28×28.
    """
    x1, y1, x2, y2 = box
    # Inset padding (clamp to image bounds)
    px1 = max(0, x1 + pad)
    py1 = max(0, y1 + pad)
    px2 = min(img.shape[1], x2 - pad)
    py2 = min(img.shape[0], y2 - pad)

    if px2 <= px1 or py2 <= py1:
        return np.zeros((28, 28), dtype=np.uint8)

    crop = img[py1:py2, px1:px2]

    # Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Threshold: ink (dark) -> white; background (light including ghost) -> black
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Resize to 28×28 for the CNN
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    return resized


def read_field(
    img: np.ndarray,
    boxes: list[Box],
) -> list[Digit]:
    """
    For each box in *boxes* (in order, left-to-right):
      1. Preprocess to 28×28 binary image (THRESH_BINARY_INV).
      2. Call predict_digit.
      3. Return the list of (digit_str, confidence) pairs.
    """
    return [predict_digit(_preprocess_box(img, box)) for box in boxes]


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

    # ── Step 3: Field assignment
    try:
        fields = assign_fields(rows)
    except ValueError as exc:
        return {
            "data":         {},
            "issues":       [str(exc)],
            "needs_review": True,
        }

    # ── Step 4: Read each field
    field_digits: dict[str, list[Digit]] = {
        name: read_field(img, boxes)
        for name, boxes in fields.items()
    }

    # ── Step 5: Build formatted strings
    result = build_results(fields, field_digits)

    # ── Step 6: Validate
    issues, needs_review = validate(result, field_digits)

    output = {
        "data":         result,
        "issues":       issues,
        "needs_review": needs_review,
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
