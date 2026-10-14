# train_kaggle.md
# Training the BIT Career Guidance Box Detector on Kaggle (GPU T4×2)

## 1. Setup — Create Kaggle Dataset

Upload `yolo_box_detector_data.zip` as a Kaggle Dataset named **bit-box-detector-data**.
The zip structure (extract to `/kaggle/working/` — files land under `box_data/`):
```
box_data/
  images/
    train/  (508 images: syn_NNNN.jpg, non-contiguous after 3 drops)
    val/    (89  images: syn_NNNN.jpg, non-contiguous after 1 drop)
  labels/
    train/  (508 label files, matching images)
    val/    (89  label files, matching images)
  data_kaggle.yaml
```

> **Path match:** `data_kaggle.yaml` sets `path: /kaggle/working/box_data`.
> Cell 1 unzips to `/kaggle/working/` → yaml lands at
> `/kaggle/working/box_data/data_kaggle.yaml`. Train images resolve to
> `/kaggle/working/box_data/images/train/`. ✅ Paths match.

## 2. Kaggle Notebook Setup

```python
# Cell 1 — unzip dataset
import os, zipfile, shutil

DATA_ZIP  = "/kaggle/input/bit-box-detector-data/yolo_box_detector_data.zip"
DATA_ROOT = "/kaggle/working/box_data"

if not os.path.exists(DATA_ROOT):
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        z.extractall("/kaggle/working/")
print("Dataset ready at", DATA_ROOT)
```

```python
# Cell 2 — install ultralytics
!pip install ultralytics==8.4.177 --quiet
```

```python
# Cell 3 — train
from ultralytics import YOLO

model = YOLO("yolov8s.pt")          # yolov8s for better accuracy on small boxes

model.train(
    data        = "/kaggle/working/box_data/data_kaggle.yaml",
    imgsz       = 1280,
    epochs      = 100,
    batch       = 8,
    workers     = 4,
    project     = "/kaggle/working/runs/box_detector",
    name        = "v1",
    cache       = "ram",
    device      = 0,          # GPU
    patience    = 30,         # early stop after 30 epochs without improvement
    save_period = 10,         # checkpoint every 10 epochs
    lr0         = 0.01,
    lrf         = 0.01,
    warmup_epochs = 3,
    close_mosaic  = 10,
)
```

```python
# Cell 4 — verify best.pt
from ultralytics import YOLO
import glob

best = glob.glob("/kaggle/working/runs/box_detector/v1/weights/best.pt")[0]
model = YOLO(best)
print(model.info())
```

## 3. Download Weights

After training, download:
- `/kaggle/working/runs/box_detector/v1/weights/best.pt`

Place it in the repo at:
- `digit_pipeline/runs/box_detector/v1/weights/best.pt`

## 4. Train / Val Split

- Split is **by index** (sequential): images syn_0000–syn_0509 → train, syn_0510–syn_0599 → val.
- **Not** by background style or warp seed — both splits draw from the same template and the
  same augmentation pool. Each image has its own independent random warp/rotation/noise, so
  there is no leakage. Val is a held-out sample of distinct augmentation instances.
- No fixed `random.seed()` is set in `make_data.py`, so re-running would give different
  augmentations but the same 85 / 15 index split.

## 5. Expected Results (597 images, YOLOv8s, 100 epochs, T4 GPU)
- Training time: ~25 min on T4
- Expected mAP50: > 0.85 on val set (all 33 boxes visible)
- mAP may be lower (~0.6–0.7) for edge-cropped images — this is intentional
- Checkpoints saved at epochs 10, 20, 30 … (use `save_period=10`)
