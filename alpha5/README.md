# Alpha5 — Detection Module

YOLO-based detection system for marine/urban waste. Covers dataset preparation,
model training, hyperparameter tuning, six inference strategies, validation with
per-class diagnostics, and an interactive GUI visualizer.

---

## Requirements

- Linux (Docker image is Ubuntu-based)
- NVIDIA GPU with at least 8 GB VRAM (AutoBatch selects batch size automatically)
- Docker with NVIDIA Container Toolkit, or Python 3.10+ for local install
- `--shm-size=8g` required for multi-worker DataLoader inside Docker

---

## Environment Setup

### Docker (recommended)

```bash
# Build
docker build -f alpha5/Dockerfile -t alpha5:latest .

# Run — mounts repo root to /ultralytics/USER inside container
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" alpha5:latest
```

All scripts use paths relative to the repo root (`alpha5/datasets/...`,
`runs/detect/...`), which resolves correctly under the volume mount.

**Specific GPU:**
```bash
docker run -it --gpus '"device=0"' --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" alpha5:latest
```

**What the image contains:**
- Base: `ultralytics/ultralytics:8.4.11`
- numpy pinned to `1.26.4` — numpy 2.x breaks albumentations
- `pandas`, `sahi` pre-installed
- `PYTHONHASHSEED=42`, `WORKDIR /ultralytics/USER`

### Local (no Docker)

```bash
pip install ultralytics pandas matplotlib tqdm psutil pyyaml opencv-python sahi
```

Optional extras:
```bash
pip install patched-yolo-infer          # patched_inference_alpha5.py
pip install basicsr realesrgan          # preprocessing_method.py --sr_method real_esrgan
```

---

## Dataset Scripts

### `datasets/scripts/img_stratifier.py`

Stratified train/val/test split that balances **instances per class**
(not just image count). Works on both flat folders and already-split datasets.

**Auto-detection**: if the input folder contains `train/`, `val/`, or `test/`
subdirs with `images/` inside, the split-folder loader is used automatically.
Force it with `--from-split`.

```bash
# Flat folder (images + .txt side-by-side)
python alpha5/datasets/scripts/img_stratifier.py /data/raw \
  --output /data/split --train 0.7 --val 0.2 --test 0.1

# Already-split dataset (re-stratify)
python alpha5/datasets/scripts/img_stratifier.py /data/split \
  --output /data/split_v2 --train 0.7 --val 0.2 --test 0.1
```

**Algorithm (greedy instance balancing):**

1. Count per-class instances in every annotated image.
2. Compute integer targets: `target_split[c] = int(total[c] * ratio)`.
3. First pass: images are assigned to train → val → test in order, skipping
   any image that would exceed its split's per-class quota.
4. Second pass: remaining unassigned images go to the split currently furthest
   below its target (scored by average fill ratio). Overflow goes to train.

**Hard negatives** (images with empty label files):

- **Split-folder mode**: each hard negative stays in its *original* split.
- **Flat-folder mode**: hard negatives are shuffled and distributed proportionally.

---

### `datasets/scripts/yolo2coco.py`

Converts a YOLO dataset (train/val/test with `images/` and `labels/`) to COCO
JSON format. Reads class names from `data.yaml`. COCO `category_id` = YOLO
class ID + 1 (COCO is 1-indexed).

```bash
python alpha5/datasets/scripts/yolo2coco.py \
  --input /data/yolo_dataset --output /data/coco_dataset
```

---

### `datasets/scripts/collapse_classes.py`

Replaces every class ID in a YOLO dataset with a single `--target_id`
(default: `0`). Useful for training a binary "trash vs background" detector
from a multi-class annotated dataset without relabelling.

```bash
python alpha5/datasets/scripts/collapse_classes.py /data/yolo_dataset \
  --output /data/binary_dataset --target_id 0
```

---

### `datasets/scripts/create_empty_labels.py`

Scans a single split directory for images missing their `.txt` label file and
creates empty ones (hard negatives).

```bash
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train --check_only
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train --dry_run
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train
```

---

## Training

### `train/train_yolo.py`

```bash
python alpha5/train/train_yolo.py data.yaml yolo26x.pt \
  --epochs 300 --batch -1 --imgsz 640 --patience 15 --device 0
```

**Resume interrupted training:**

```bash
python alpha5/train/train_yolo.py data.yaml \
  runs/detect/train/<run_name>/weights/last.pt \
  --resume --batch <original_batch>
```

`--resume` loads all hyperparameters from the checkpoint (epochs, optimizer
state, scheduler). Pass `--batch` explicitly to avoid AutoBatch re-running
and computing a different value than the original run (e.g. if GPU free memory
differs between sessions).

| Argument | Default | Description |
|----------|---------|-------------|
| `data` | required | Path to `data.yaml` |
| `model` | required | Weights spec (`yolo26x.pt` or path to `last.pt`) |
| `--epochs` | 300 | Training epochs |
| `--batch` | -1 | Batch size; `-1` = AutoBatch |
| `--imgsz` | 640 | Training image size |
| `--workers` | 4 | DataLoader workers |
| `--patience` | 15 | Early stopping patience |
| `--device` | `0` | `0`, `cpu`, `0,1,2,3` for DDP |
| `--project` | `/ultralytics/USER/runs/detect/train` | Output root |
| `--name` | timestamp | Run subfolder name |
| `--optimizer` | `auto` | `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, `auto` |
| `--hyperparams` | None | YAML file with extra kwargs merged into `model.train()` |
| `--resume` | False | Resume from checkpoint |
| `--verbose` | False | Print CUDA/PyTorch info at startup |

---

### `train/hyperparam_yolo_tunning.py`

Bayesian hyperparameter search using Ultralytics `model.tune()`. Best
hyperparameters saved under `runs/detect/tune/`.

```bash
python alpha5/train/hyperparam_yolo_tunning.py data.yaml yolo26x.pt 30 50 \
  --device 0 --name tune_exp --tune_kwargs hparams_seed.yaml
```

| Positional | Description |
|------------|-------------|
| `data` | Path to `data.yaml` |
| `model` | Weights spec |
| `epochs` | Epochs per trial |
| `iterations` | Number of tuning trials |

---

### `tests/experiments/best_epoch.py`

Reads Ultralytics `results.csv` and prints the row with the highest
`mAP@0.50-95`. Accepts multiple column name variants for compatibility
across Ultralytics versions.

```bash
python alpha5/tests/experiments/best_epoch.py \
  runs/detect/train/exp/results.csv
```

---

## Validation & Evaluation

### `tests/experiments/val_yolo.py`

Full validation with per-class diagnostics, optional CSV export, bar chart,
and predicted-image export.

```bash
python alpha5/tests/experiments/val_yolo.py data.yaml best.pt \
  --per_class_csv --plot_classes --plots --predict_val --concat
```

**Always printed:** global mAP@0.50 / mAP@0.50-95 / mAP@0.75 / P / R / F1,
plus per-class table with `contrib_%`.

**`contrib_%` — how it works:**

```
contrib_i = AP_class_i / sum(AP_all_classes) * 100
```

- All contrib values sum to **100%**.
- Equal share = `100 / N_classes` (≈14.3% for 7 classes).
- Class above line → boosting mAP. Class below → dragging mAP down.

**Optional outputs:**

| Flag | Output |
|------|--------|
| `--per_class_csv` | `<run_dir>/metrics_per_class.csv` |
| `--plot_classes` | `<run_dir>/metrics_per_class.png` — mAP/F1 bars + contrib% bars |
| `--plots` | Confusion matrix + Ultralytics standard plots |
| `--predict_val` | Annotated images + `.txt` labels for val set |
| `--concat` | Side-by-side original/prediction JPEGs |

---

## Inference Methods

All methods write annotated JPEGs to `--out_dir`. All accept `--device`
(default `cuda:0`) and `--conf` (default 0.25).

| Script | Strategy | Fusion | Speed | Best for |
|--------|----------|--------|-------|----------|
| `inference.py` | Single-scale | — | 1.0× | Baseline, benchmarking |
| `static_slices.py` | Tiled | NMS | 1.5–2.5× | High-res, simple setup |
| `inference_tiled.py` | Tiled | WBF or NMS + dedup | 1.5–2.5× | High-res, trash deprioritization |
| `multi_scale_ensemble.py` | Multi-scale | NMS | 3.0–4.0× | Variable object sizes |
| `tta.py` | TTA (5 augs) | NMS | 4.0–6.0× | Orientation/lighting variability |
| `preprocessing_method.py` | SR + single-scale | — | 1.1–1.3× | Low-quality imagery |
| `hybrid_pipeline.py` | Full + crops | WBF (5 stages) | 2.5–3.5× | Best quality, research |
| `sahi_dir.py` | SAHI sliced | SAHI internal | 1.5–2.5× | External library alternative |
| `patched_inference_alpha5.py` | Patched | NMS | 1.5–2.5× | patched-yolo-infer library |

---

### `inference.py` — Basic

```bash
python alpha5/tests/experiments/inference.py images/ best.pt output/ \
  --conf 0.25 --imgsz 640
```

Logs wall-clock time and RSS memory delta per image. Baseline for benchmarking.

---

### `static_slices.py` — Tiled (NMS)

```bash
python alpha5/tests/experiments/static_slices.py images/ best.pt \
  --crops 4 --overlap 0.2 --iou 0.45
```

`--crops` must be even. Internally forms a sqrt(N) × ceil(N/sqrt(N)) grid.
Max 8 crops recommended.

---

### `inference_tiled.py` — Tiled with WBF + Deduplication

```bash
python alpha5/tests/experiments/inference_tiled.py images/ best.pt \
  --crops 6 --overlap 0.2 --fusion wbf --iou 0.5 \
  --iou_dedup 0.5 --prioritize_specific --trash_id 7
```

`--fusion wbf`: averages overlapping boxes weighted by confidence instead of
discarding. `--prioritize_specific`: specific class beats generic `trash` (class 7)
on overlap regardless of confidence.

---

### `multi_scale_ensemble.py` — Multi-Scale

```bash
python alpha5/tests/experiments/multi_scale_ensemble.py images/ best.pt \
  --scales 640 960 1280 --nms_thresh 0.5
```

---

### `tta.py` — Test-Time Augmentation

5 augmentations per image: original, horizontal flip, vertical flip, both flips,
scale 1.1×. Boxes are un-transformed before fusion.

```bash
python alpha5/tests/experiments/tta.py images/ best.pt \
  --conf 0.25 --tta_iou 0.5 --imgsz 640
```

---

### `preprocessing_method.py` — Super-Resolution Preprocessing

| `--sr_method` | Technique | Extra deps |
|---|---|---|
| `clahe` | CLAHE on LAB L-channel | none |
| `unsharp` | Unsharp mask | none |
| `opencv_dnn` | ESPCN/EDSR/FSRCNN ×2 | `ESPCN_x2.pb` model file |
| `real_esrgan` | Real-ESRGAN ×2 | `basicsr`, `realesrgan` |

```bash
python alpha5/tests/experiments/preprocessing_method.py images/ best.pt \
  --sr_method clahe --conf 0.25
```

---

### `hybrid_pipeline.py` — Hybrid 5-Stage Pipeline

```bash
python alpha5/tests/experiments/hybrid_pipeline.py images/ best.pt \
  --crops 6 --overlap 0.2 \
  --high_iou 0.85 --suspect_iou 0.3 --merge_iou 0.5 \
  --save_intermediate --draw_grid
```

**Stages:** full-image inference → crops + WBF → smart filter → merge → confidence filter.

**Smart filter (Stage 3):** for each crop detection, compare max IoU vs same-class full detections:

| IoU range | Decision |
|---|---|
| `>= high_iou` (0.85) | Same object seen by both — **keep** |
| `< suspect_iou` (0.3) | New small object — **keep** |
| Between both | Likely crop fragment — **discard** |

`--save_intermediate` saves images at stages 1, 2, and 3 for debugging.

---

### `sahi_dir.py` — SAHI Sliced Inference

```bash
pip install sahi
python alpha5/tests/experiments/sahi_dir.py images/ best.pt output/ \
  --slice_height 320 --slice_width 320 \
  --overlap_height_ratio 0.2 --overlap_width_ratio 0.2
```

---

### `patched_inference_alpha5.py` — patched-yolo-infer

```bash
pip install patched-yolo-infer
python alpha5/tests/experiments/patched_inference_alpha5.py images/ best.pt \
  --patch_size 640 --overlap 0.25 --save_comparison
```

---

## Shared Utilities

### `wbf_utils.py`

| Function | Description |
|----------|-------------|
| `weighted_boxes_fusion` | BFS clustering per class. Fused box = score-weighted average of corners. Score = cluster max. |
| `greedy_nms_classwise` | Per-class greedy NMS by descending confidence. Returns keep-indices. |
| `deduplicate_detections` | On IoU overlap: specific class beats `trash` if `prioritize_non_trash=True`, else highest confidence wins. |

**WBF vs NMS:** NMS discards all but highest-confidence box in a cluster.
WBF merges by averaging — smoother positions, no loss of confidence signal.

---

### `crop_utils.py`

| Component | Description |
|-----------|-------------|
| `UniformCrops` | Computes sqrt(N)×ceil(N/sqrt(N)) grid. Stride = `cell × (1 − overlap)`. |
| `draw_crop_grid` | Draws crop rectangles (numbered) on copy of image. |
| `iter_images` | Lists `.jpg/.jpeg/.png` from file or directory, sorted. |

---

### Utility scripts

| Script | Description |
|--------|-------------|
| `crop_maker.py` | Extract and save crops without inference. `--save_crops --draw_grid`. |
| `pair_concat.py` | Side-by-side concatenation of two image directories by alphabetical order. |
| `test_dup.py` | Visual diagnostic: saves 3 versions (raw / dedup / dedup+trash-deprioritized) for one image. |

---

## GUI Visualizer

```bash
cd alpha5/tests/visualizer
python run_visualizer.py
```

Interactive side-by-side comparison of inference methods. Built with Tkinter.
Inference runs in a background thread so the UI stays responsive.

**Features:** load model + image via file dialog, zoom-to-cursor (mousewheel),
click-drag pan, double-click to fit, dark theme. Method dropdown populated from
`inference_methods.py` registry.

**Add a method:** implement `BaseMethod` from `alpha5_base.py`, register in
`inference_methods.py`.

---

## Directory Structure

```
alpha5/
├── Dockerfile
├── train/
│   ├── train_yolo.py               — Training entry point
│   └── hyperparam_yolo_tunning.py  — Bayesian hyperparameter tuning
├── datasets/scripts/
│   ├── img_stratifier.py           — Stratified train/val/test split
│   ├── yolo2coco.py                — YOLO → COCO format conversion
│   ├── collapse_classes.py         — Merge all classes into one (binary detector)
│   └── create_empty_labels.py      — Create hard-negative label files
├── tests/experiments/
│   ├── val_yolo.py                 — Validation + per-class metrics
│   ├── best_epoch.py               — Find best epoch from training CSV
│   ├── inference.py                — Basic inference with timing/memory log
│   ├── static_slices.py            — Tiled inference (NMS)
│   ├── inference_tiled.py          — Tiled inference (WBF + deduplication)
│   ├── multi_scale_ensemble.py     — Multi-scale ensemble
│   ├── tta.py                      — Test-Time Augmentation (5 augs)
│   ├── preprocessing_method.py     — SR preprocessing (CLAHE/unsharp/DNN/ESRGAN)
│   ├── hybrid_pipeline.py          — 5-stage hybrid pipeline
│   ├── sahi_dir.py                 — SAHI sliced inference
│   ├── patched_inference_alpha5.py — patched-yolo-infer wrapper
│   ├── crop_maker.py               — Extract crops without inference
│   ├── pair_concat.py              — Side-by-side image concatenation
│   ├── test_dup.py                 — Visual deduplication diagnostic
│   ├── crop_utils.py               — UniformCrops, draw_crop_grid, iter_images
│   └── wbf_utils.py                — WBF, greedy NMS, deduplicate_detections
└── tests/visualizer/
    ├── run_visualizer.py           — Launch the GUI
    ├── alpha5_visualizer.py        — Tkinter GUI implementation
    ├── alpha5_base.py              — Base inference wrapper
    ├── inference_methods.py        — Method registry
    ├── utils.py                    — GUI utilities
    └── test_methods.py             — Method smoke tests
```
