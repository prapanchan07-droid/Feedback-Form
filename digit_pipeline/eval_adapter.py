"""
eval_adapter.py
───────────────
Evidence A / B / C: measures how well the Keras digit model performs when
used through the predict_digit adapter on realistic synthetic boxes.

Evidence A — Synthetic box builder
    Draws each MNIST test digit as dark ink in a ~25 x (28-45 px) box
    (random height to mimic form variability), adds:
      * a faint dotted grey border (BGR 180)
      * a ghost "8" in light grey (BGR 220)
    Then passes each box through the teammate's _preprocess_box
    (3 px inset, threshold 120, THRESH_BINARY_INV, resize 28x28).

Evidence B -- Accuracy comparison
    1. Keras model on teammate's crops with NO extra normalisation (raw uint8 /255)
    2. Keras model with adapter's normalize_crop (bounding-rect, CoM, etc.)
    3. Keras model on clean MNIST 28x28 (baseline)
    Also: blank-box accuracy (all-empty + border-only + ghost-only boxes)

Evidence C -- Save 40 visual examples to scratch/
    Before: the synthetic box (BGR)
    After:  the crop after _preprocess_box (uint8 28x28)
    Adapted: after normalize_crop (float32 28x28 -> saved as uint8 for viewing)

Usage
-----
    CNN\\venv\\Scripts\\python.exe digit_pipeline/eval_adapter.py

Outputs
-------
    scratch/eval_adapter/   -- 40 triplet images + summary table
    stdout                  -- accuracy table

MNIST loading
-------------
    Uses pure urllib + gzip (no torch/torchvision dependency) to avoid
    the numpy>=2 conflict introduced by modern torch wheels.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from digit_pipeline.predict_digit import normalize_crop  # noqa: E402

# ── Import teammate's preprocessing (READ-ONLY) ───────────────────────────────
# We import and call _preprocess_box directly; we do not copy its logic.
from digit_pipeline.form_merge import _preprocess_box  # noqa: E402

# ── Output directory ──────────────────────────────────────────────────────────
SCRATCH_DIR = ROOT / "scratch" / "eval_adapter"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 500          # MNIST test digits to use for B1/B2
N_VISUAL  = 40           # crops to save for Evidence C
CONF_THRESH = 0.35       # below this → treat as low-confidence (not wrong)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic box builder (Evidence A)
# ══════════════════════════════════════════════════════════════════════════════

def _build_synthetic_box(
    digit_img_28: np.ndarray,
    label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a BGR form-box image that imitates a real Bannari Amman form cell.

    Steps
    ─────
    1. Create a white box of width=25, height=random(28,46).
    2. Add a dotted grey border (every other pixel along each edge, grey=180).
    3. Overlay a faint ghost "8" in grey=220 (imitating the guide digit).
    4. Place the MNIST digit (inverted so it appears as dark ink on white).
    5. Return the BGR image.
    """
    box_w = 25
    box_h = int(rng.integers(28, 46))

    # White background
    box = np.full((box_h, box_w, 3), 255, dtype=np.uint8)

    # Dotted grey border
    for x in range(0, box_w, 2):
        box[0, x]         = 180
        box[box_h - 1, x] = 180
    for y in range(0, box_h, 2):
        box[y, 0]         = 180
        box[y, box_w - 1] = 180

    # Ghost "8" centred — very faint (220 grey on white = very light)
    ghost = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    gh    = min(box_h, 22)
    cv2.putText(ghost, "8", (3, gh), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    # Blend ghost into box
    mask = (ghost < 240)
    box[mask] = ghost[mask]

    # MNIST digit: digit_img_28 is white-on-black (0..255); we want dark ink
    # on white box. Invert, then scale to fit in ~20×20 area of the box.
    digit_inv = 255 - digit_img_28     # black digit on white

    # Resize digit to fit the box (aspect-preserving, max ~80% of box dims)
    max_h = max(1, int(box_h * 0.80))
    max_w = max(1, int(box_w * 0.80))
    scale = min(max_h / 28.0, max_w / 28.0)
    nw    = max(1, int(28 * scale))
    nh    = max(1, int(28 * scale))
    digit_resized = cv2.resize(digit_inv, (nw, nh), interpolation=cv2.INTER_AREA)

    # Paste with a small random jitter (±2 px)
    x0 = (box_w - nw) // 2 + int(rng.integers(-2, 3))
    y0 = (box_h - nh) // 2 + int(rng.integers(-2, 3))
    x0 = max(1, min(box_w - nw - 1, x0))
    y0 = max(1, min(box_h - nh - 1, y0))

    digit_bgr = cv2.cvtColor(digit_resized, cv2.COLOR_GRAY2BGR)
    roi = box[y0:y0 + nh, x0:x0 + nw]
    # Blend: keep the darker pixel (ink is darker than ghost/border)
    box[y0:y0 + nh, x0:x0 + nw] = np.minimum(roi, digit_bgr)

    return box


def _build_blank_box(
    box_type: str,   # "empty" | "border_only" | "ghost_only"
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a box with no real digit for blank-detection testing."""
    box_w = 25
    box_h = int(rng.integers(28, 46))
    box = np.full((box_h, box_w, 3), 255, dtype=np.uint8)

    if box_type in ("border_only", "ghost_only"):
        for x in range(0, box_w, 2):
            box[0, x]         = 180
            box[box_h - 1, x] = 180
        for y in range(0, box_h, 2):
            box[y, 0]         = 180
            box[y, box_w - 1] = 180

    if box_type == "ghost_only":
        ghost = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
        cv2.putText(ghost, "8", (3, min(box_h, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        mask = (ghost < 240)
        box[mask] = ghost[mask]

    return box


# ══════════════════════════════════════════════════════════════════════════════
# Keras model helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_keras_model() -> Any:
    import tensorflow as tf
    model_path = os.environ.get(
        "DIGIT_MODEL_PATH",
        str(ROOT / "CNN" / "outputs" / "digit_cnn.keras")
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Keras model not found at: {model_path}")
    print(f"Loading Keras model from: {model_path}")
    return tf.keras.models.load_model(model_path)


def _batch_predict_raw(model: Any, crops_u8: list[np.ndarray]) -> list[tuple[int, float]]:
    """
    Option B1: raw — just divide by 255 and reshape; no geometry normalisation.
    """
    batch = np.array(
        [c.astype(np.float32) / 255.0 for c in crops_u8],
        dtype=np.float32
    ).reshape(-1, 28, 28, 1)
    probs = model.predict(batch, verbose=0)
    return [(int(np.argmax(p)), float(p.max())) for p in probs]


def _batch_predict_adapted(model: Any, crops_u8: list[np.ndarray]) -> list[tuple[int, float]]:
    """
    Option B2: with adapter's normalize_crop (bounding-rect, CoM, aspect resize).
    """
    tensors = [normalize_crop(c) for c in crops_u8]
    batch   = np.concatenate(tensors, axis=0)
    probs   = model.predict(batch, verbose=0)
    return [(int(np.argmax(p)), float(p.max())) for p in probs]


# ══════════════════════════════════════════════════════════════════════════════
# Evidence B helpers
# ══════════════════════════════════════════════════════════════════════════════

def _accuracy_report(preds: list[tuple[int, float]], labels: list[int], name: str) -> dict:
    correct = sum(p == l for (p, _), l in zip(preds, labels))
    # Count predictions where confidence < CONF_THRESH
    low_conf = sum(conf < CONF_THRESH for _, conf in preds)
    n = len(labels)
    acc = correct / n if n else 0
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Samples  : {n}")
    print(f"  Correct  : {correct}  ({acc * 100:.2f}%)")
    print(f"  Low-conf : {low_conf}  ({low_conf/n*100:.1f}%  below conf={CONF_THRESH})")
    return {"name": name, "n": n, "correct": correct, "accuracy": acc, "low_conf": low_conf}


# ══════════════════════════════════════════════════════════════════════════════
# Pure MNIST loader (urllib + gzip, no torch/torchvision)
# ══════════════════════════════════════════════════════════════════════════════

_MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
_MNIST_FILES = {
    "images": "t10k-images-idx3-ubyte.gz",
    "labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_mnist(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download MNIST test set if not cached; return (images, labels)."""
    import gzip
    import struct
    import urllib.request  # noqa: PLC0415

    cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(fname: str) -> bytes:
        dest = cache_dir / fname
        if not dest.exists():
            url = f"{_MNIST_BASE}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
        with gzip.open(str(dest), "rb") as f:
            return f.read()

    # Images: magic 2051, n_images, rows, cols, then raw bytes
    img_raw = _fetch(_MNIST_FILES["images"])
    magic, n, rows, cols = struct.unpack_from(">IIII", img_raw, 0)
    images = np.frombuffer(img_raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)

    # Labels: magic 2049, n_labels, then raw bytes
    lbl_raw = _fetch(_MNIST_FILES["labels"])
    magic_l, n_l = struct.unpack_from(">II", lbl_raw, 0)
    labels = np.frombuffer(lbl_raw, dtype=np.uint8, offset=8).astype(np.int64)

    return images, labels


def _load_mnist_test(
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Load the MNIST test split (10 000 images) and return n_samples random ones.
    Each image is a uint8 (28, 28) array with white ink on black background,
    matching the convention used by _preprocess_box.
    """
    cache_dir = ROOT / "scratch" / "_mnist_cache"
    all_images, all_labels = _download_mnist(cache_dir)

    # MNIST is originally black ink on white; invert so white ink on black
    all_images = 255 - all_images

    idxs = rng.choice(len(all_images), size=n_samples, replace=False)
    imgs  = [all_images[i] for i in idxs]
    lbls  = [int(all_labels[i]) for i in idxs]
    return imgs, lbls


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    rng = np.random.default_rng(42)

    # ── Load MNIST test set (pure urllib + gzip, no torch dependency) ───────────
    print("Loading MNIST test set (pure urllib download) ...")
    mnist_imgs, mnist_labels = _load_mnist_test(N_SAMPLES, rng)
    print(f"Loaded {N_SAMPLES} MNIST test samples.")

    # ── Load Keras model ──────────────────────────────────────────────────────
    model = _load_keras_model()

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence A: build synthetic boxes, pass through _preprocess_box
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVIDENCE A — Synthetic box generation + _preprocess_box")
    print("=" * 60)

    syn_boxes   = []   # raw BGR synthetic boxes
    preproc_crops = []  # uint8 (28,28) after _preprocess_box

    for img_u8 in mnist_imgs:
        box_bgr = _build_synthetic_box(img_u8, 0, rng)   # label unused here
        crop    = _preprocess_box(box_bgr, [0, 0, box_bgr.shape[1], box_bgr.shape[0]])
        syn_boxes.append(box_bgr)
        preproc_crops.append(crop)

    print(f"  Built {len(preproc_crops)} synthetic boxes.")
    print(f"  Example crop dtype: {preproc_crops[0].dtype}, "
          f"shape: {preproc_crops[0].shape}, "
          f"max: {preproc_crops[0].max()}, min: {preproc_crops[0].min()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence B1: Keras on teammate's crops, NO extra normalisation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVIDENCE B — Accuracy comparison")
    print("=" * 60)

    t0 = time.time()
    preds_raw = _batch_predict_raw(model, preproc_crops)
    t1 = time.time()
    r1 = _accuracy_report(preds_raw, mnist_labels,
                          "B1: Keras on teammate crops — NO re-normalisation (raw /255)")
    print(f"  Inference time: {t1-t0:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence B2: Keras with adapter's normalize_crop
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    preds_adapted = _batch_predict_adapted(model, preproc_crops)
    t1 = time.time()
    r2 = _accuracy_report(preds_adapted, mnist_labels,
                          "B2: Keras on teammate crops — WITH adapter normalize_crop")
    print(f"  Inference time: {t1-t0:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence B3: Clean MNIST 28×28 baseline (as-is)
    # ══════════════════════════════════════════════════════════════════════════
    clean_batch = np.array(
        [img.astype(np.float32) / 255.0 for img in mnist_imgs],
        dtype=np.float32
    ).reshape(-1, 28, 28, 1)
    t0 = time.time()
    clean_probs = model.predict(clean_batch, verbose=0)
    t1 = time.time()
    preds_clean = [(int(np.argmax(p)), float(p.max())) for p in clean_probs]
    r3 = _accuracy_report(preds_clean, mnist_labels,
                          "B3: Keras on CLEAN MNIST 28×28 (baseline)")
    print(f"  Inference time: {t1-t0:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence B4: Blank-box detection accuracy
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  B4: Blank-box accuracy")

    from digit_pipeline.predict_digit import predict_digit  # noqa: E402

    blank_types   = ["empty"] * 60 + ["border_only"] * 60 + ["ghost_only"] * 60
    blank_correct = 0
    blank_total   = len(blank_types)

    for bt in blank_types:
        box_bgr = _build_blank_box(bt, rng)
        crop    = _preprocess_box(box_bgr, [0, 0, box_bgr.shape[1], box_bgr.shape[0]])
        digit, _ = predict_digit(crop)
        if digit == "":
            blank_correct += 1

    blank_acc = blank_correct / blank_total
    print(f"  Blank types: empty×60, border_only×60, ghost_only×60")
    print(f"  Correctly detected as blank: {blank_correct}/{blank_total} "
          f"({blank_acc * 100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # Evidence C: Save 40 visual examples
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVIDENCE C — Saving 40 visual examples")
    print("=" * 60)

    n_save = min(N_VISUAL, len(syn_boxes))
    save_idxs = rng.choice(len(syn_boxes), size=n_save, replace=False)

    for rank, si in enumerate(save_idxs):
        raw_box  = syn_boxes[si]                           # BGR synthetic box
        pre_crop = preproc_crops[si]                       # uint8 (28,28) from _preprocess_box
        adapted  = normalize_crop(pre_crop)                 # (1,28,28,1) float32
        adapted_u8 = (adapted[0, :, :, 0] * 255).astype(np.uint8)  # (28,28) for display

        # Stack side-by-side: raw_box | pre_crop (resized to match height) | adapted
        h_target = 56
        # Panel 1: raw box (upscale for visibility)
        scale1 = h_target / raw_box.shape[0]
        p1 = cv2.resize(raw_box, (int(raw_box.shape[1] * scale1), h_target))

        # Panel 2: pre_crop (28×28 → 56×56)
        p2_gray = cv2.resize(pre_crop, (56, h_target), interpolation=cv2.INTER_NEAREST)
        p2 = cv2.cvtColor(p2_gray, cv2.COLOR_GRAY2BGR)

        # Panel 3: adapted (28×28 → 56×56)
        p3_gray = cv2.resize(adapted_u8, (56, h_target), interpolation=cv2.INTER_NEAREST)
        p3 = cv2.cvtColor(p3_gray, cv2.COLOR_GRAY2BGR)

        # Dividers (2 px white vertical line)
        div = np.full((h_target, 2, 3), 200, dtype=np.uint8)

        strip = np.hstack([p1, div, p2, div, p3])

        # Label with truth and both predictions
        lbl = mnist_labels[si]
        pred_raw, conf_raw       = preds_raw[si]
        pred_ada, conf_ada       = preds_adapted[si]

        # Annotate below strip
        info_h = 20
        info   = np.full((info_h, strip.shape[1], 3), 50, dtype=np.uint8)
        cv2.putText(info,
                    f"gt={lbl} | raw={pred_raw}({conf_raw:.2f}) | ada={pred_ada}({conf_ada:.2f})",
                    (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0) if pred_ada == lbl else (0, 80, 255), 1)

        final = np.vstack([strip, info])
        fname = SCRATCH_DIR / f"crop_{rank:02d}_gt{lbl}_raw{pred_raw}_ada{pred_ada}.png"
        cv2.imwrite(str(fname), final)

    print(f"  Saved {n_save} example images to: {SCRATCH_DIR}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary table
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Condition':<52}  {'Accuracy':>8}  {'Low-conf':>8}")
    print("─" * 72)
    for r in [r1, r2, r3]:
        print(f"{r['name'][:52]:<52}  {r['accuracy']*100:>7.2f}%  {r['low_conf']:>8}")
    print(f"{'B4: Blank detection accuracy':<52}  {blank_acc*100:>7.2f}%  {'─':>8}")
    print("─" * 72)

    # Write CSV for reference
    import pandas as pd  # noqa: PLC0415
    rows = [r1, r2, r3, {"name": "B4 Blank detection", "n": blank_total,
                          "correct": blank_correct, "accuracy": blank_acc, "low_conf": 0}]
    pd.DataFrame(rows).to_csv(SCRATCH_DIR / "accuracy_results.csv", index=False)
    print(f"\nResults CSV saved to: {SCRATCH_DIR / 'accuracy_results.csv'}")


if __name__ == "__main__":
    main()
