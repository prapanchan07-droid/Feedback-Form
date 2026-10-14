# digit_pipeline – Handwritten Digit Form Reader
## Bannari Amman Institute of Technology · Feedback Form

A fully self-contained pipeline for **detecting digit boxes** with YOLOv8 and
**merging / validating** extracted field values from the 33-box digit form.

---

## Folder Structure

```
Feedback-Form/
├── bannari_amman_form_template.pdf   ← original 1-page form
├── bannari_amman_form_template.png   ← existing 300-DPI render (optional)
│
└── digit_pipeline/
    ├── render_template.py   # Step 1 – PDF → PNG at 300 DPI
    ├── label_boxes.py       # Step 2 – GUI box labelling → boxes.json
    ├── make_data.py         # Step 3 – Synthetic dataset generation
    ├── data.yaml            # Step 4 – YOLOv8 dataset config
    ├── form_merge.py        # Task 2 – Field merging, validation, batch CSV
    └── README.md            # This file
```

---

## Prerequisites

```powershell
pip install ultralytics opencv-python-headless numpy pymupdf pillow torchvision
```

> **GPU recommended** for training.  
> CPU-only inference works fine for the merging pipeline.

---

## Quick-Start – Run Each Step in Order

### Step 0 · (optional) Create the digit_pipeline folder and cd into it

```powershell
cd "C:\Users\Sachin Akash\Desktop\project\Feedback-Form"
mkdir digit_pipeline
cd digit_pipeline
```

---

### Step 1 · Render the PDF to PNG at 300 DPI

```powershell
python render_template.py --pdf ../bannari_amman_form_template.pdf --dpi 300 --out template_300dpi.png
```

Output: `digit_pipeline/template_300dpi.png`  
*(The repo already contains `bannari_amman_form_template.png` – you may copy/rename that instead.)*

---

### Step 2 · Label the 33 Digit Boxes Once (interactive)

```powershell
python label_boxes.py --img template_300dpi.png --out boxes.json
```

A window opens with the template.  

**Controls:**
| Action | Key |
|--------|-----|
| Drag to draw ROI | Mouse |
| Confirm ROI | `SPACE` or `ENTER` |
| Cancel current ROI (re-try) | `ESC` |
| Undo last accepted | `u` |
| Save & quit | `q` |

Follow the on-screen counter — label boxes in this exact order:

| # | Group | Count |
|---|-------|-------|
| 1–8 | Date (DD MM YYYY) | 8 |
| 9–18 | Student Mobile | 10 |
| 19–28 | Parent Mobile | 10 |
| 29–33 | Cut-off (ABC.DE, **no decimal box**) | 5 |

> ⚠️ **Skip** the 0-9 reference strip in the top-right corner.  
> ⚠️ **Skip** the printed decimal point — it is not a writable box.

Output: `digit_pipeline/boxes.json`

---

### Step 3 · Generate 600 Synthetic Training Images

```powershell
python make_data.py --template template_300dpi.png --boxes boxes.json --out data/ --n 600
```

MNIST digits are downloaded automatically to `./mnist_data/` the first time.

Output:
```
data/
  images/train/*.jpg   (510 images)
  images/val/*.jpg     (90 images)
  labels/train/*.txt
  labels/val/*.txt
```

---

### Step 4 · Train YOLOv8s (box detector)

```powershell
yolo train model=yolov8s.pt data=data.yaml imgsz=1280 epochs=100 batch=8 ^
     project=runs/box_detector name=v1 cache=ram
```

Best weights are saved to `runs/box_detector/v1/weights/best.pt`.

---

### Step 5 · Process a Single Form Scan

```powershell
python form_merge.py single path/to/scan.jpg --weights runs/box_detector/v1/weights/best.pt
```

With debug visualisation:
```powershell
python form_merge.py single path/to/scan.jpg --weights runs/box_detector/v1/weights/best.pt --debug
```

Outputs:
- `<stem>_result.json` – extracted data, issues, needs_review flag
- `debug_fields.jpg` *(when --debug)* – colour-coded field overlay

---

### Step 6 · Batch Process a Folder

```powershell
python form_merge.py batch path/to/scans_folder/ --weights runs/box_detector/v1/weights/best.pt --out results.csv
```

Output: `results.csv` with columns:
`filename, date, student_mobile, parent_mobile, cutoff, needs_review, issues`

---

## Swapping in the Real CNN

Open `form_merge.py` and find the section:

```python
# ══════════════════════════════════════════════════════════════════════════════
# ── CNN STUB ──
```

Replace the body of `predict_digit(crop_28x28)` with your real model call:

```python
def predict_digit(crop_28x28: np.ndarray) -> tuple[str, float]:
    tensor = your_preprocess(crop_28x28)        # normalise to your model's input
    logits = your_model(tensor)
    probs  = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    digit = str(pred.item()) if conf.item() > 0.35 else ""
    return digit, conf.item()
```

No other changes needed — the rest of the pipeline calls this function.

---

## Fine-Tuning on Real Scans (30–50 labelled images)

When the synthetic-only model misses or misdetects boxes on real scanned forms:

### 1. Collect 30–50 real scans and label them

Use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg)  
to draw bounding boxes around every digit box (class `box`).  
Export in **YOLOv8 format**.

### 2. Create `data_real.yaml`

```yaml
path:  data_real
train: images/train
val:   images/val
nc:    1
names: [box]
```

Mix real + synthetic data for best results:

```yaml
# data_mixed.yaml
path:  .
train:
  - data/images/train
  - data_real/images/train
val:   data_real/images/val
nc:    1
names: [box]
```

### 3. Fine-tune from the synthetic checkpoint

```powershell
yolo train model=runs/box_detector/v1/weights/best.pt ^
     data=data_real.yaml imgsz=1280 epochs=30 batch=4 ^
     freeze=10 ^
     project=runs/box_detector name=v1_real
```

- `freeze=10` freezes the YOLOv8s backbone (first 10 layers) so only the
  detection head adapts to the real domain — avoids overfitting on 30–50 images.
- Reduce to `epochs=15` and `batch=2` if training on CPU.

### 4. Evaluate

```powershell
yolo val model=runs/box_detector/v1_real/weights/best.pt data=data_real.yaml imgsz=1280
```

Target: **mAP50 ≥ 0.90** on the real-scan validation set.

---

## Form Layout Reference

```
┌──────────────────────────────────────────────────────────────────┐
│  [0-9 Reference Strip – TOP RIGHT – IGNORED]                     │
│                                                                  │
│  Row 1 – DATE (8 boxes)                                          │
│  ┌──┐ ┌──┐  ┌──┐ ┌──┐  ┌──┐ ┌──┐ ┌──┐ ┌──┐                    │
│  │D1│ │D2│  │M1│ │M2│  │Y1│ │Y2│ │Y3│ │Y4│                    │
│  └──┘ └──┘  └──┘ └──┘  └──┘ └──┘ └──┘ └──┘                    │
│                                                                  │
│  Row 2 – STUDENT MOBILE (10) ──gap── PARENT MOBILE (10)         │
│  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐   ← student         │
│  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐   ← parent          │
│                                                                  │
│  Row 3 – CUTOFF (5 boxes; decimal is printed between 3rd & 4th) │
│  ┌─┐ ┌─┐ ┌─┐  ●  ┌─┐ ┌─┐                                       │
│  │A│ │B│ │C│  .  │D│ │E│                                        │
│  └─┘ └─┘ └─┘     └─┘ └─┘                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Validation Rules

| Field | Rule |
|-------|------|
| date | Must parse as a real Gregorian date (DDMMYYYY) |
| student_mobile | Matches `[6-9]\d{9}` (Indian mobile format) |
| parent_mobile | Matches `[6-9]\d{9}` |
| cutoff | Numeric value in [0, 200] |
| any digit | Confidence < 0.80 → flagged in `issues` |

Forms with any issue have `needs_review: true` in the JSON output.
