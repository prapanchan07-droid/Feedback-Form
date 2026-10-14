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

# ─────────────────────────────────────────────────────────── config constants

CANVAS_PAD  = 40          # extra padding added around the warped form
BOX_FILL_P  = 0.70        # probability a box gets a digit
TRAIN_RATIO = 0.85
YOLO_CLASS  = 0           # single class: "box"

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
) -> None:
    """Paste *digit_patch* (dark on white, uint8) into *box* on *img* in-place."""
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    # Leave a small inner margin so the digit doesn't touch the border
    margin = max(2, min(bw, bh) // 8)
    iw, ih = bw - 2 * margin, bh - 2 * margin
    if iw < 4 or ih < 4:
        return
    patch = cv2.resize(digit_patch, (iw, ih), interpolation=cv2.INTER_AREA)
    # Dark ink (< 128) is written; bright background (~255) is transparent
    roi = img[y1 + margin: y2 - margin, x1 + margin: x2 - margin]
    if roi.shape[:2] != patch.shape[:2]:
        return
    # Multiply: keep existing texture where patch is bright (paper), darken where digit
    alpha = (255 - patch.astype(np.float32)) / 255.0    # 1 where ink, 0 where paper
    if len(roi.shape) == 3:
        alpha = alpha[:, :, np.newaxis]
    ink_colour = np.full_like(roi, 20, dtype=np.float32)  # near-black ink
    blended = roi.astype(np.float32) * (1 - alpha) + ink_colour * alpha
    img[y1 + margin: y2 - margin, x1 + margin: x2 - margin] = np.clip(blended, 0, 255).astype(np.uint8)


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

    print(f"[make_data] Template {template.shape[1]}×{template.shape[0]}, "
          f"{len(boxes)} boxes, generating {n} images …")

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

        # ── Fill ~70 % of boxes with MNIST digits
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
                _paste_digit(img, box, patch)

        # ── Geometric augmentations (transform both image AND boxes)
        img, cur_boxes = _random_scale(img, cur_boxes)
        img, cur_boxes = _random_rotate(img, cur_boxes)
        img, cur_boxes = _random_perspective(img, cur_boxes)

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
