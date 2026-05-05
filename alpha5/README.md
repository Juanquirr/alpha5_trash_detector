# alpha5 — Detection Module

YOLO26-based waste detection with six configurable inference strategies and an interactive Tkinter GUI for visual comparison.

---

## Requirements

| Mode | What you need |
|------|--------------|
| Docker (recommended) | Docker + NVIDIA GPU runtime, 8+ GB VRAM |
| Native | Python 3.8+, `ultralytics`, `opencv-python`, `Pillow`, `sahi`, `numpy==1.26.4` |

> **Note:** numpy 2.x breaks `albumentations`. The Dockerfile pins `numpy==1.26.4`.

---

## Docker Setup

Build from the **repo root**:

```bash
docker build -f alpha5/Dockerfile -t alpha5:latest .
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" alpha5:latest
```

The working directory inside the container is `/ultralytics/USER`. Your local repo is mounted there, so edits and outputs are visible from both sides.

---

## Dataset Preparation

### Instance-stratified split

Splits a flat or pre-split YOLO dataset preserving per-class instance ratios across train/val/test.

```bash
python alpha5/datasets/scripts/img_stratifier.py /path/to/data \
  --output /path/to/output --train 0.7 --val 0.2 --test 0.1
```

### Other dataset scripts

| Script | Purpose |
|--------|---------|
| `yolo2coco.py` | Convert YOLO `.txt` labels → COCO JSON |
| `collapse_classes.py` | Merge multiple class IDs into one |
| `create_empty_labels.py` | Generate empty `.txt` for unannotated images |

---

## Training

```bash
python alpha5/train/train_yolo.py /path/to/data.yaml yolo26x.pt \
  --epochs 300 --batch -1 --imgsz 640 --patience 15 --device 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 300 | Training epochs |
| `--batch` | -1 | Batch size (`-1` = AutoBatch) |
| `--imgsz` | 640 | Input resolution |
| `--patience` | 15 | Early-stop patience (epochs without mAP gain) |
| `--device` | `0` | GPU index, `cpu`, or `0,1,2,3` for multi-GPU DDP |
| `--project` | `/ultralytics/USER/runs/detect/train` | Output root |
| `--name` | timestamp | Experiment subfolder name |

Results land in `runs/detect/train/{name}/`: weights, plots, metrics CSV.

### Hyperparameter Tuning

Bayesian optimisation over the Ultralytics hyperparameter search space:

```bash
python alpha5/train/hyperparam_yolo_tunning.py /path/to/data.yaml yolo26x.pt <iterations> <epochs_per_trial>
# e.g.:
python alpha5/train/hyperparam_yolo_tunning.py data.yaml yolo26x.pt 50 20
```

---

## Validation & Per-Class Analysis

`val_yolo.py` runs Ultralytics validation and always prints a full per-class breakdown immediately after global metrics.

### Basic run

```bash
python alpha5/tests/experiments/val_yolo.py data.yaml runs/train/exp/weights/best.pt
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--imgsz` | 640 | Validation image size |
| `--batch` | 16 | Batch size |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.35 | IoU threshold |
| `--device` | auto | GPU index or `cpu` |
| `--plots` | off | Save confusion matrix + PR curves |
| `--per_class_csv` | off | Export `metrics_per_class.csv` |
| `--plot_classes` | off | Save `metrics_per_class.png` bar chart |
| `--predict_val` | off | Run prediction pass on val images after val |
| `--concat` | off | Save original \| prediction side-by-side JPEGs |
| `--project` | `full_validation_run` | Output root dir |
| `--name` | `val_full` | Subfolder under project |

### Terminal output

After global metrics, the script always prints:

```
===== PER-CLASS METRICS =====

 ID  Class                     P       R      F1    mAP50  mAP50-95  Contrib%
--------------------------------------------------------------------
  0  plastic_bottle        0.8123  0.7654  0.7881   0.8200    0.5421    14.23%
  1  glass                 0.6200  0.5800  0.5993   0.6100    0.3800     9.97%
  2  can                   0.7900  0.8100  0.7999   0.8300    0.5600    14.70%
  ...
```

### How each metric is computed

Ultralytics returns per-class arrays in `metrics.box`, but they are **indexed by position in `ap_class_index`**, not by raw class ID. A class absent from the val set has no entry in `ap_class_index` and gets all zeros.

The alignment step:

```python
idx_map = {int(cid): i for i, cid in enumerate(metrics.box.ap_class_index)}
# then: p = metrics.box.p[idx_map[class_id]]
```

From that, the reported columns are:

| Column | Formula |
|--------|---------|
| `Precision` | `metrics.box.p[i]` — TP / (TP + FP) at default conf |
| `Recall` | `metrics.box.r[i]` — TP / (TP + FN) at default conf |
| `F1` | `2 · P · R / (P + R)` |
| `mAP50` | `metrics.box.ap50[i]` — area under PR curve at IoU 0.50 |
| `mAP50-95` | `metrics.box.ap[i]` — mean of AP at IoU 0.50:0.05:0.95 |
| `Contrib%` | `class mAP50-95 / global mAP50-95 × 100` |

**`Contrib%`** is the most diagnostic column: a class at 12.5% is pulling its equal weight (100% / 8 classes). Below that — the class is dragging global mAP down. Above — it carries more than its share.

### Optional outputs

**`--per_class_csv`** writes `{project}/{name}/metrics_per_class.csv` with all seven columns.

**`--plot_classes`** writes `{project}/{name}/metrics_per_class.png`:
- Top panel — grouped bars: mAP@0.50 / mAP@0.50-95 / F1 per class, with value labels
- Bottom panel — `Contrib%` bars + red dashed equal-share reference line

Example with all outputs enabled:

```bash
python alpha5/tests/experiments/val_yolo.py data.yaml best.pt \
  --plots --per_class_csv --plot_classes \
  --project runs/val --name exp1
# outputs:
#   runs/val/exp1/metrics_per_class.csv
#   runs/val/exp1/metrics_per_class.png
#   runs/val/exp1/confusion_matrix.png  (from --plots)
```

---

## Inference Methods

Six strategies exposed under `alpha5/tests/experiments/`:

| Method | Relative speed | Small objects | Large objects | Best for |
|--------|---------------|---------------|---------------|----------|
| Basic | 1.0× | Fair | Excellent | Real-time baseline |
| Tiled | 1.5–2.5× | Good | Good | High-res / dense scenes |
| MultiScale | 3.0–4.0× | Excellent | Good | Variable object sizes |
| TTA | 4.0–6.0× | Fair | Good | Orientation/lighting variability |
| SuperRes | 1.1–1.3× | Good | Good | Low-quality imagery |
| **Hybrid** | **2.5–3.5×** | **Excellent** | **Excellent** | **Research / validation** |

Entry points inside the experiments folder: `inference.py` (basic), `inference_tiled.py`, `multi_scale_ensemble.py`, `tta.py`, `preprocessing_method.py` (SuperRes), `hybrid_pipeline.py`.

WBF (Weighted Box Fusion) merging is in `wbf_utils.py` and used by Hybrid + MultiScale.

---

## GUI Visualizer

Interactive side-by-side comparison of all six methods on any image or folder.

```bash
# from alpha5/tests/visualizer/
python run_visualizer.py
```

Features:
- Zoom-to-cursor (mousewheel) + drag pan on any result pane
- Double-click to fit image to pane
- Collapsible method panels
- Adjustable confidence and IoU thresholds
- Load model `.pt` via file dialog

Requires a display (not headless). On the server, run via X11 forwarding or RDP.

---

## Directory Structure

```
alpha5/
├── Dockerfile
├── datasets/
│   └── scripts/
│       ├── img_stratifier.py        Instance-stratified split
│       ├── yolo2coco.py             YOLO → COCO conversion
│       ├── collapse_classes.py      Merge class IDs
│       └── create_empty_labels.py   Empty labels for clean images
├── train/
│   ├── train_yolo.py                Training entry point
│   └── hyperparam_yolo_tunning.py   Bayesian hyperparameter search
└── tests/
    ├── experiments/
    │   ├── inference.py             Basic inference
    │   ├── inference_tiled.py       SAHI tiled inference
    │   ├── multi_scale_ensemble.py  MultiScale + WBF
    │   ├── tta.py                   Test-Time Augmentation
    │   ├── preprocessing_method.py  SuperRes upscaling
    │   ├── hybrid_pipeline.py       Hybrid (Tiled + MultiScale + WBF)
    │   ├── wbf_utils.py             Weighted Box Fusion helpers
    │   └── val_yolo.py              Validation on a YOLO dataset split
    └── visualizer/
        ├── run_visualizer.py        GUI launcher
        ├── alpha5_visualizer.py     Main Tkinter app
        ├── inference_methods.py     Method registry for the GUI
        └── utils.py                 Draw helpers
```
