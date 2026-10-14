"""
make_data.py
────────────
Step 3 of the box-detection pipeline.

Generates 600 synthetic training images from the blank template PNG.
Each image has:
  - Random perspective warp, rotation (±5°), scale (0.85-1.0)
  - Padded canvas with random background grey
  - Gaussian blur + salt-pepper noise + brightness/contrast jitter
  - ~70% of boxes filled with a random MNIST-style digit (dark ink)
  - Corresponding YOLO-format label files (class 0, normalised xywh)
  - 85 / 15 train / val split

Prerequisites:
    pip install torch torchvision Pillow numpy opencv-python

Usage:
    python make_data.py --template template_300dpi.png --boxes boxes.json
                        --out data/ --n 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from torchvision.datasets import MNIST
    import torchvision.transforms as T
    _HAS_MNIST = True
except ImportError:
    _HAS_MNIST = False

try:
    from digit_pipeline.auto_boxes import validate_box_sizes
except ImportError:
    try:
        from auto_boxes import validate_box_sizes
    except ImportError:
        validate_box_sizes = None  # graceful fallback if running standalone

# ─────────────────────────────────────────────────────────── config constants

CANVAS_PAD        = 40     # extra padding added around the warped form
BOX_FILL_P        = 0.70   # probability a box gets a digit
TRAIN_RATIO       = 0.85
YOLO_CLASS        = 0      # single class: "box"
OVERFLOW_DIGIT_P  = 0.15   # prob a filled box gets an overflow-offset digit
EDGE_CROP_P       = 0.10   # prob image has one page-edge randomly cropped

# ──────────────────────────────────────────────────────────────── MNIST loader

def _load_mnist_digits(mnist_root: str = "./mnist_data") -> dict[int, list[np.ndarray]]:
    """
    Download MNIST (or load from cache) and return a dict mapping
    digit -> list of 28×28 uint8 grayscale arrays (dark digit on white).
    """
    if not _HAS_MNIST:
        print("[make_data] torchvision not available – using synthetic digit patches.")
        return {}

    ds = MNIST(root=mnist_root, train=True, download=True)
    by_digit: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
    for img, label in ds:
        arr = np.array(img)           # 28×28, 0=black bg, 255=white digit (MNIST convention)
        arr = 255 - arr               # invert: dark digit on white background
        by_digit[label].append(arr)
    return by_digit


def _make_synthetic_digit_patch(digit: int, size: tuple[int, int]) -> np.ndarray:
    """
    Fallback: draw the digit with PIL and return a grayscale patch (dark on white).
    """
    w, h = size
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    draw.text((w // 5, h // 10), str(digit), fill=20)
    return np.array(img)


# ──────────────────────────────────────────────────────── augmentation helpers

def _random_perspective(
    img: np.ndarray,
    boxes: list[list[int]],
    max_shift_frac: float = 0.03,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Apply a mild random perspective warp.
    Returns the warped image and transformed boxes ([x1,y1,x2,y2]).
    """
    h, w = img.shape[:2]
    shift = lambda: random.uniform(-max_shift_frac, max_shift_frac)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * shift(), h * shift()],
        [w + w * shift(), h * shift()],
        [w + w * shift(), h + h * shift()],
        [w * shift(), h + h * shift()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(200, 200, 200))

    new_boxes: list[list[int]] = []
    for x1, y1, x2, y2 in boxes:
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        nx1 = int(np.min(transformed[:, 0]))
        ny1 = int(np.min(transformed[:, 1]))
        nx2 = int(np.max(transformed[:, 0]))
        ny2 = int(np.max(transformed[:, 1]))
        new_boxes.append([
            max(0, nx1), max(0, ny1),
            min(w - 1, nx2), min(h - 1, ny2),
        ])

    return warped, new_boxes


def _random_rotate(
    img: np.ndarray,
    boxes: list[list[int]],
    max_deg: float = 5.0,
) -> tuple[np.ndarray, list[list[int]]]:
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(200, 200, 200))

    new_boxes: list[list[int]] = []
    for x1, y1, x2, y2 in boxes:
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        ones = np.ones((4, 1), dtype=np.float32)
        c_h = np.hstack([corners, ones])
        t = (M @ c_h.T).T
        nx1 = int(np.min(t[:, 0]))
        ny1 = int(np.min(t[:, 1]))
        nx2 = int(np.max(t[:, 0]))
        ny2 = int(np.max(t[:, 1]))
        new_boxes.append([
            max(0, nx1), max(0, ny1),
            min(w - 1, nx2), min(h - 1, ny2),
        ])
    return rotated, new_boxes


def _random_scale(
    img: np.ndarray,
    boxes: list[list[int]],
    scale_range: tuple[float, float] = (0.85, 1.0),
) -> tuple[np.ndarray, list[list[int]]]:
    s = random.uniform(*scale_range)
    h, w = img.shape[:2]
    nh, nw = int(h * s), int(w * s)
    scaled = cv2.resize(img, (nw, nh))

    # Pad back to original size with random grey background
    bg_val = random.randint(180, 230)
    canvas = np.full((h, w, 3) if len(img.shape) == 3 else (h, w),
                     bg_val, dtype=np.uint8)
    canvas[:nh, :nw] = scaled

    new_boxes = [
        [int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)]
        for x1, y1, x2, y2 in boxes
    ]
    return canvas, new_boxes


def _apply_noise_and_blur(img: np.ndarray) -> np.ndarray:
    # Gaussian blur
    ksize = random.choice([1, 1, 3, 3, 5])
    if ksize > 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Salt-and-pepper noise
    if random.random() < 0.5:
        density = random.uniform(0.001, 0.01)
        mask = np.random.rand(*img.shape[:2])
        img[mask < density / 2] = 0
        img[mask > 1 - density / 2] = 255

    return img


def _brightness_contrast(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.85, 1.15)   # contrast
    beta  = random.randint(-20, 20)      # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _paste_digit(
    img: np.ndarray,
    box: list[int],
    digit_patch: np.ndarray,
    overflow: bool = False,
) -> None:
    """
    Paste *digit_patch* (dark on white, uint8) into *box* on *img* in-place.
    If *overflow* is True the digit is randomly offset so it crosses the cell
    border (simulates sloppy handwriting that overflows the box).
    """
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    margin = max(2, min(bw, bh) // 8)
    iw, ih = bw - 2 * margin, bh - 2 * margin
    if iw < 4 or ih < 4:
        return
    patch = cv2.resize(digit_patch, (iw, ih), interpolation=cv2.INTER_AREA)

    if overflow:
        # Random offset: 10-40% of cell dimension in a random direction
        max_dx = int(bw * random.uniform(0.10, 0.40))
        max_dy = int(bh * random.uniform(0.10, 0.40))
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
    else:
        dx, dy = 0, 0

    ry1 = y1 + margin + dy
    ry2 = y2 - margin + dy
    rx1 = x1 + margin + dx
    rx2 = x2 - margin + dx

    img_h, img_w = img.shape[:2]
    # Clamp paste region to image bounds
    ry1c, ry2c = max(0, ry1), min(img_h, ry2)
    rx1c, rx2c = max(0, rx1), min(img_w, rx2)
    py1c = ry1c - ry1   # corresponding patch slice start
    py2c = py1c + (ry2c - ry1c)
    px1c = rx1c - rx1
    px2c = px1c + (rx2c - rx1c)

    if ry2c <= ry1c or rx2c <= rx1c or py2c <= py1c or px2c <= px1c:
        return
    patch_crop = patch[py1c:py2c, px1c:px2c]
    roi = img[ry1c:ry2c, rx1c:rx2c]
    if roi.shape[:2] != patch_crop.shape[:2]:
        return

    alpha = (255 - patch_crop.astype(np.float32)) / 255.0
    if len(roi.shape) == 3:
        alpha = alpha[:, :, np.newaxis]
    ink_colour = np.full_like(roi, 20, dtype=np.float32)
    blended = roi.astype(np.float32) * (1 - alpha) + ink_colour * alpha
    img[ry1c:ry2c, rx1c:rx2c] = np.clip(blended, 0, 255).astype(np.uint8)


def _random_edge_crop(
    img: np.ndarray,
    boxes: list[list[int]],
    crop_frac_range: tuple[float, float] = (0.05, 0.20),
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Crop one random edge of the image by 5-20%, simulating a partially
    scanned page edge.  Boxes that fall outside the remaining area are
    clamped; labels with zero area after clamping are kept (they become
    very small and the NMS in training handles them).
    """
    h, w = img.shape[:2]
    side  = random.choice(['left', 'right', 'top', 'bottom'])
    frac  = random.uniform(*crop_frac_range)
    if side == 'left':
        cut = int(w * frac)
        img = img[:, cut:]
        boxes = [[max(0, x1-cut), y1, max(0, x2-cut), y2] for x1,y1,x2,y2 in boxes]
    elif side == 'right':
        cut = int(w * frac)
        img = img[:, :w-cut]
        boxes = [[x1, y1, min(w-cut, x2), y2] for x1,y1,x2,y2 in boxes]
    elif side == 'top':
        cut = int(h * frac)
        img = img[cut:, :]
        boxes = [[x1, max(0, y1-cut), x2, max(0, y2-cut)] for x1,y1,x2,y2 in boxes]
    else:  # bottom
        cut = int(h * frac)
        img = img[:h-cut, :]
        boxes = [[x1, y1, x2, min(h-cut, y2)] for x1,y1,x2,y2 in boxes]
    return img, boxes


# ────────────────────────────────────────────────────────── label writer helper

def _to_yolo(box: list[int], img_w: int, img_h: int) -> str:
    """Convert [x1,y1,x2,y2] to YOLO normalised 'class cx cy w h'."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{YOLO_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ─────────────────────────────────────────────────────────────────────── main

def generate_dataset(
    template_path: str,
    boxes_path: str,
    out_dir: str,
    n: int = 600,
    mnist_root: str = "./mnist_data",
) -> None:
    template = cv2.imread(template_path)
    if template is None:
        sys.exit(f"[ERROR] Cannot read template: {template_path}")

    with open(boxes_path) as f:
        boxes: list[list[int]] = json.load(f)  # list of [x1,y1,x2,y2]

    # ── Guard: fail loudly if boxes.json has size inconsistencies
    if validate_box_sizes is not None:
        if not validate_box_sizes(boxes):
            sys.exit("[make_data] ABORT: boxes.json failed 15% size check. "
                     "Re-run auto_boxes.py or fix boxes.json.")
    else:
        print("[make_data] WARNING: validate_box_sizes not importable; "
              "skipping size check.")

    print(f"[make_data] Template {template.shape[1]}x{template.shape[0]}, "
          f"{len(boxes)} boxes, generating {n} images ...")

    # ── MNIST cache
    mnist_cache = _load_mnist_digits(mnist_root)

    # ── Output directory structure
    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_n = int(n * TRAIN_RATIO)

    for i in range(n):
        split = "train" if i < train_n else "val"
        stem  = f"syn_{i:04d}"

        img = template.copy()
        cur_boxes = [list(b) for b in boxes]   # deep copy

        # ── Fill ~70 % of boxes with MNIST digits (15 % of those overflow)
        for b_idx, box in enumerate(cur_boxes):
            if random.random() < BOX_FILL_P:
                digit = random.randint(0, 9)
                if mnist_cache:
                    patches = mnist_cache[digit]
                    patch = patches[random.randint(0, len(patches) - 1)].copy()
                else:
                    bw = box[2] - box[0]
                    bh = box[3] - box[1]
                    patch = _make_synthetic_digit_patch(digit, (bw, bh))
                overflow = random.random() < OVERFLOW_DIGIT_P
                _paste_digit(img, box, patch, overflow=overflow)

        # ── Geometric augmentations (transform both image AND boxes)
        img, cur_boxes = _random_scale(img, cur_boxes)
        img, cur_boxes = _random_rotate(img, cur_boxes)
        img, cur_boxes = _random_perspective(img, cur_boxes)

        # ── Page-edge crop (~10 % of images)
        if random.random() < EDGE_CROP_P:
            img, cur_boxes = _random_edge_crop(img, cur_boxes)

        # ── Photometric augmentations
        img = _brightness_contrast(img)
        img = _apply_noise_and_blur(img)

        h, w = img.shape[:2]

        # ── Write image
        img_path = out / "images" / split / f"{stem}.jpg"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # ── Write YOLO labels (only boxes that are still inside the image)
        lbl_path = out / "labels" / split / f"{stem}.txt"
        lines = []
        for b in cur_boxes:
            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(_to_yolo(b, w, h))
        lbl_path.write_text("\n".join(lines))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} done …")

    print(f"[make_data] Dataset written to '{out_dir}'")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic YOLO training data.")
    p.add_argument("--template",   default="template_300dpi.png")
    p.add_argument("--boxes",      default="boxes.json")
    p.add_argument("--out",        default="data/")
    p.add_argument("--n",          type=int, default=600)
    p.add_argument("--mnist_root", default="./mnist_data")
    args = p.parse_args()
    generate_dataset(args.template, args.boxes, args.out, args.n, args.mnist_root)


if __name__ == "__main__":
    main()
