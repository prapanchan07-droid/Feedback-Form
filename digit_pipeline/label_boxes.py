"""
label_boxes.py
──────────────
Step 2 of the box-detection pipeline.

Interactive OpenCV script: open the rendered template PNG, let the user
draw ROIs on the 33 digit boxes (skipping the top-right reference strip),
and save them to boxes.json as a list of [x1, y1, x2, y2] coordinates.

Controls:
  - Drag to select each box, press SPACE or ENTER to confirm each ROI.
  - Press ESC to cancel selection of the current ROI.
  - Press 'q' after all ROIs are drawn to save and exit.
  - Press 'u' to undo the last accepted box.

Usage:
    python label_boxes.py --img template_300dpi.png --out boxes.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------- helpers

LABEL_ORDER = [
    # Row 1: Date  (8 boxes)
    "date_d1", "date_d2", "date_m1", "date_m2",
    "date_y1", "date_y2", "date_y3", "date_y4",
    # Row 2: Student mobile  (10 boxes)
    "smob_0", "smob_1", "smob_2", "smob_3", "smob_4",
    "smob_5", "smob_6", "smob_7", "smob_8", "smob_9",
    # Row 2: Parent mobile  (10 boxes)
    "pmob_0", "pmob_1", "pmob_2", "pmob_3", "pmob_4",
    "pmob_5", "pmob_6", "pmob_7", "pmob_8", "pmob_9",
    # Row 3: Cutoff  (5 boxes – skip the printed decimal point)
    "co_a", "co_b", "co_c", "co_d", "co_e",
]

TOTAL_BOXES = len(LABEL_ORDER)   # 33


def draw_overlay(
    base_img: np.ndarray,
    accepted: list[dict],
    current_label: str,
) -> np.ndarray:
    """Return a copy of base_img with accepted boxes and next-label instructions."""
    vis = base_img.copy()
    colors = [
        (0, 200, 0),    # date        – green
        (200, 80, 0),   # student mob – blue-ish
        (0, 80, 200),   # parent mob  – orange
        (180, 0, 180),  # cutoff      – purple
    ]

    for box in accepted:
        idx   = box["idx"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        col   = colors[min(idx // 8, 3)]  # rough grouping by row
        if idx >= 8 and idx < 18:
            col = colors[1]
        elif idx >= 18 and idx < 28:
            col = colors[2]
        elif idx >= 28:
            col = colors[3]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, box["label"], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # Instructions panel
    cv2.rectangle(vis, (0, 0), (620, 50), (30, 30, 30), -1)
    msg = (
        f"({len(accepted)}/{TOTAL_BOXES})  Next: '{current_label}'  "
        "[Drag+SPACE=confirm | u=undo | q=save+quit]"
    )
    cv2.putText(vis, msg, (8, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 180), 1, cv2.LINE_AA)
    return vis


def run_labeller(img_path: str, out_path: str) -> None:
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {img_path}")

    # Scale down for display if very large (keep coords in original space)
    MAX_DISP_W, MAX_DISP_H = 1600, 1000
    oh, ow = img.shape[:2]
    scale = min(MAX_DISP_W / ow, MAX_DISP_H / oh, 1.0)
    display_img = cv2.resize(img, (int(ow * scale), int(oh * scale)),
                             interpolation=cv2.INTER_AREA)

    accepted: list[dict] = []

    win = "Box Labeller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_img.shape[1], display_img.shape[0])

    while len(accepted) < TOTAL_BOXES:
        idx   = len(accepted)
        label = LABEL_ORDER[idx]

        overlay = draw_overlay(display_img, accepted, label)
        cv2.imshow(win, overlay)

        # selectROI returns (x, y, w, h) in display-image space
        roi = cv2.selectROI(win, overlay, showCrosshair=True, fromCenter=False)
        rx, ry, rw, rh = roi

        # Press ESC → rw==0 and rh==0 → skip / re-try
        if rw == 0 or rh == 0:
            print(f"  Skipped ROI for '{label}' (zero area). Repeat.")
            continue

        # Map back to original image space
        x1 = int(rx / scale)
        y1 = int(ry / scale)
        x2 = int((rx + rw) / scale)
        y2 = int((ry + rh) / scale)

        accepted.append({"idx": idx, "label": label,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        print(f"  [{idx+1}/{TOTAL_BOXES}] '{label}' -> [{x1},{y1},{x2},{y2}]")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("u") and accepted:
            removed = accepted.pop()
            print(f"  Undid: '{removed['label']}'")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Save as a simple list (index-ordered) of [x1, y1, x2, y2]
    out: list[list[int]] = []
    for box in sorted(accepted, key=lambda b: b["idx"]):
        out.append([box["x1"], box["y1"], box["x2"], box["y2"]])

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[label_boxes] Saved {len(out)} boxes -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively label 33 digit boxes.")
    parser.add_argument("--img", default="template_300dpi.png",
                        help="Template PNG rendered by render_template.py.")
    parser.add_argument("--out", default="boxes.json",
                        help="Output JSON file.")
    args = parser.parse_args()

    if not Path(args.img).exists():
        sys.exit(f"[ERROR] Image not found: {args.img}  (run render_template.py first)")

    run_labeller(args.img, args.out)


if __name__ == "__main__":
    main()
