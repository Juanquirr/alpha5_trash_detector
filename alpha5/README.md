# Alpha5 — Detection Module

YOLO-based detection system for marine/urban waste. Covers dataset preparation,
model training, hyperparameter tuning, six inference strategies, validation with
per-class diagnostics, and an interactive GUI visualizer.

---

## Directory Structure

```
alpha5/
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
│   ├── static_slices.py            — Tiled inference (NMS, simple version)
│   ├── inference_tiled.py          — Tiled inference (NMS or WBF + deduplication)
│   ├── multi_scale_ensemble.py     — Multi-scale ensemble inference
│   ├── tta.py                      — Test-Time Augmentation
│   ├── preprocessing_method.py     — Super-Resolution preprocessing inference
│   ├── hybrid_pipeline.py          — 5-stage hybrid pipeline (best quality)
│   ├── sahi_dir.py                 — SAHI sliced inference (external library)
│   ├── patched_inference_alpha5.py — patched-yolo-infer library wrapper
│   ├── crop_maker.py               — Extract and save crops without inference
│   ├── pair_concat.py              — Side-by-side image concatenation
│   ├── test_dup.py                 — Visual test of deduplication logic
│   ├── crop_utils.py               — Shared: UniformCrops, iter_images
│   └── wbf_utils.py                — Shared: WBF, greedy NMS, deduplication
└── tests/visualizer/
    ├── run_visualizer.py           — Launch the GUI
    ├── alpha5_visualizer.py        — GUI implementation (Tkinter)
    ├── alpha5_base.py              — Base inference wrapper
    ├── inference_methods.py        — Method registry for GUI
    ├── utils.py                    — GUI utilities
    └── test_methods.py             — Method smoke tests
```

---

## Training

### `train/train_yolo.py`

Trains a YOLO model with an `on_fit_epoch_end` callback that logs mAP50,
best mAP50, and patience counter to stdout each epoch. Supports AutoBatch,
multi-GPU DDP, and external hyperparameter YAML.

```bash
python alpha5/train/train_yolo.py data.yaml yolo26x.pt \
  --epochs 300 --batch -1 --imgsz 640 --patience 15 --device 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `data` | required | Path to `data.yaml` |
| `model` | required | Weights spec (`yolo26x.pt` or `/path/to/last.pt`) |
| `--epochs` | 300 | Training epochs |
| `--batch` | -1 | Batch size; `-1` = AutoBatch |
| `--imgsz` | 640 | Training image size |
| `--workers` | 4 | DataLoader workers |
| `--patience` | 15 | Early stopping patience |
| `--device` | `0` | `0`, `cpu`, `0,1,2,3` for DDP |
| `--project` | `/ultralytics/USER/runs/detect/train` | Output root |
| `--name` | timestamp | Run subfolder name |
| `--optimizer` | `auto` | `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, `auto` |
| `--hyperparams` | None | YAML file with extra hyperparameters (merged into `model.train()`) |
| `--resume` | False | Resume from `last.pt` |
| `--verbose` | False | Print CUDA/PyTorch info at startup |

**Hyperparameter YAML** (`--hyperparams`): any Ultralytics `model.train()` kwarg.
`close_mosaic` is auto-cast to `int`. Reserved keys (`data`, `epochs`, etc.)
are ignored to avoid conflicts.

---

### `train/hyperparam_yolo_tunning.py`

Bayesian hyperparameter search using Ultralytics `model.tune()`. Runs
`iterations` trials of `epochs` each. Best hyperparameters saved in the run
directory under `runs/detect/tune/`.

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

`--tune_kwargs`: optional YAML with starting-point hyperparameter values
(e.g. `lr0`, `lrf`, `momentum`). Reserved keys are automatically stripped
to avoid overriding the tuner.

---

### `tests/experiments/best_epoch.py`

Reads Ultralytics `results.csv` and prints the row with the highest
`mAP@0.50-95`. Useful for locating `weights/epoch_N.pt` after training.

```bash
python alpha5/tests/experiments/best_epoch.py runs/detect/train/exp/results.csv
```

Accepts multiple column name variants (`metrics/mAP50-95(B)`,
`metrics/mAP_0.5:0.95`, `val/mAP50-95`, etc.) for compatibility across
Ultralytics versions.

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
  Their per-split counts are printed and they are copied as-is.
- **Flat-folder mode**: hard negatives are shuffled and distributed
  proportionally according to `--train/--val/--test` ratios.

| Argument | Default | Description |
|----------|---------|-------------|
| `input_folder` | required | Flat folder or split-folder root |
| `--output` / `-o` | `<input>/balanced_by_instances` | Output root |
| `--train` | 0.7 | Instance fraction for train |
| `--val` | 0.2 | Instance fraction for val |
| `--test` | 0.1 | Instance fraction for test |
| `--from-split` | False | Force split-folder loader |

---

### `datasets/scripts/yolo2coco.py`

Converts a YOLO dataset (train/val/test with `images/` and `labels/`) to COCO
JSON format. Reads class names from `data.yaml`. COCO `category_id` = YOLO
class ID + 1 (COCO is 1-indexed). Hard negatives (images with empty labels)
are included in the COCO `images` list with no annotations.

```bash
python alpha5/datasets/scripts/yolo2coco.py \
  --input /data/yolo_dataset --output /data/coco_dataset
```

Output structure:
```
<output>/
  images/train/   images/val/   images/test/
  annotations/instances_train.json
  annotations/instances_val.json
  annotations/instances_test.json
  data.yaml
```

---

### `datasets/scripts/collapse_classes.py`

Replaces every class ID in a YOLO dataset with a single `--target_id`
(default: `0`). Useful for training a binary "trash vs background" detector
from a multi-class annotated dataset without relabelling.

- Hard negative label files (empty) are preserved unchanged.
- Malformed annotation lines (< 5 fields or non-integer class ID) are skipped
  with a warning counter.

```bash
python alpha5/datasets/scripts/collapse_classes.py /data/yolo_dataset \
  --output /data/binary_dataset --target_id 0
```

---

### `datasets/scripts/create_empty_labels.py`

Scans a single split directory for images missing their `.txt` label file and
creates empty ones (hard negatives). Useful after adding background-only images
to an existing dataset.

```bash
# Check structure only
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train --check_only

# Dry run (show what would be created)
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train --dry_run

# Actually create
python alpha5/datasets/scripts/create_empty_labels.py /data/split/train
```

Reports: total images, label files, empty labels (negatives), non-empty labels,
and missing labels before and after.

---

## Validation & Evaluation

### `tests/experiments/val_yolo.py`

Full validation with per-class diagnostics, optional CSV export, bar chart,
and predicted-image export.

```bash
python alpha5/tests/experiments/val_yolo.py data.yaml best.pt \
  --per_class_csv --plot_classes --plots \
  --predict_val --concat
```

**Always printed:**

- **Global metrics**: mAP@0.50, mAP@0.50-95, mAP@0.75, Precision, Recall, F1
- **Per-class table**: P, R, F1, mAP50, mAP50-95, `contrib_%` for every class

**`contrib_%` — how it works:**

```
contrib_i = AP_class_i / sum(AP_all_classes) * 100
```

- All contrib values sum to **100%**.
- Equal share = `100 / N_classes` (12.5% for 8 classes).
- Class above equal-share line → boosting mAP.
- Class below equal-share line → dragging mAP down.
- This is the primary diagnostic for identifying which waste categories the
  model struggles with.

**Optional outputs:**

| Flag | Output |
|------|--------|
| `--per_class_csv` | `<run_dir>/metrics_per_class.csv` — one row per class |
| `--plot_classes` | `<run_dir>/metrics_per_class.png` — two panels: mAP/F1 bars + contrib% bars with equal-share line |
| `--plots` | Confusion matrix + Ultralytics standard plots |
| `--predict_val` | Run `model.predict()` on val images; saves annotated images + label `.txt` |
| `--concat` | Side-by-side original/prediction JPEGs in `predictions_val_concat/` |

| Key argument | Default | Description |
|--------------|---------|-------------|
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.35 | IoU threshold |
| `--batch` | 16 | Validation batch size |
| `--imgsz` | 640 | Image size |
| `--val_images` | from yaml | Override val images directory |
| `--project` | `full_validation_run` | Output root |
| `--name` | `val_full` | Run subfolder |

---

## Inference Methods

Six methods exist, from simplest to most complex. All write annotated JPEGs to
`--out_dir`. All accept `--device` (default `cuda:0`) and `--conf` (default 0.25).

---

### `inference.py` — Basic Inference

Single-scale `model.predict()` per image. Logs wall-clock time and RSS memory
delta for every image. Baseline for benchmarking.

```bash
python alpha5/tests/experiments/inference.py images/ best.pt output/ \
  --conf 0.25 --imgsz 640 --device cuda
```

---

### `static_slices.py` — Tiled Inference (NMS, simple)

Splits each image into a uniform overlapping grid, runs inference on each crop,
maps detections back to global coordinates, applies class-wise greedy NMS.
Simpler variant without WBF or deduplication.

```bash
python alpha5/tests/experiments/static_slices.py images/ best.pt \
  --crops 4 --overlap 0.2 --iou 0.45 --out_dir tiled_out
```

**`--crops`** must be even and positive. Internally forms a √N × ⌈N/√N⌉ grid.
Max 8 crops recommended; beyond that, per-crop resolution drops too low.

---

### `inference_tiled.py` — Tiled Inference with WBF + Deduplication

Extended tiled inference adding:
- **Fusion choice** (`--fusion wbf` or `nms`): WBF averages boxes in overlapping
  clusters weighted by confidence rather than simply discarding lower-scored boxes.
- **Deduplication** (`wbf_utils.deduplicate_detections`): removes redundant
  detections across crops using IoU threshold.
- **Trash deprioritization** (`--prioritize_specific`): when a specific class
  (e.g. `plastic_bottle`) overlaps with generic `trash` (class 7), the specific
  class wins regardless of confidence.

```bash
python alpha5/tests/experiments/inference_tiled.py images/ best.pt \
  --crops 6 --overlap 0.2 --fusion wbf --iou 0.5 \
  --iou_dedup 0.5 --prioritize_specific --trash_id 7
```

Use `--save_crops` to inspect per-crop annotated images.
Use `--draw_grid` to visualize the crop grid on the original image.

---

### `multi_scale_ensemble.py` — Multi-Scale Ensemble

Runs `model.predict()` at multiple image sizes on the same image, concatenates
all detections, and applies NMS to fuse them.

```bash
python alpha5/tests/experiments/multi_scale_ensemble.py images/ best.pt \
  --scales 640 960 1280 --nms_thresh 0.5
```

Default scales: `640 960 1280`. Adding `1920` helps very large objects;
removing `640` speeds up at cost of small object recall. Each additional scale
multiplies inference time roughly linearly.

---

### `tta.py` — Test-Time Augmentation

Generates 5 augmented versions of each image, runs inference on all, reverses
the geometric transforms on detected boxes, then fuses with NMS.

| Augmentation | Transform | Reverse |
|---|---|---|
| Original | — | — |
| Horizontal flip | `cv2.flip(img, 1)` | `x' = W - x` |
| Vertical flip | `cv2.flip(img, 0)` | `y' = H - y` |
| Both flips | `cv2.flip(img, -1)` | both axes |
| Scale 1.1× | `cv2.resize(..., fx=1.1)` | `box / 1.1` |

```bash
python alpha5/tests/experiments/tta.py images/ best.pt \
  --conf 0.25 --tta_iou 0.5 --imgsz 640
```

`--tta_iou`: IoU threshold used when fusing the 5 augmented predictions.
Typically set equal to or slightly higher than the standard `--iou`.

---

### `preprocessing_method.py` — Super-Resolution Preprocessing

Applies an image enhancement step before YOLO inference to improve recall on
low-quality or low-contrast imagery.

| `--sr_method` | Technique | Dependencies |
|---|---|---|
| `clahe` | CLAHE on LAB L-channel (contrast adaptive HE) | OpenCV only |
| `unsharp` | Unsharp mask (Gaussian blur subtract) | OpenCV only |
| `opencv_dnn` | ESPCN/EDSR/FSRCNN x2 upscale | `ESPCN_x2.pb` model file |
| `real_esrgan` | Real-ESRGAN x2 upscale | `basicsr`, `realesrgan` |

`clahe` and `unsharp` are zero-dependency and run on CPU/GPU. The DNN methods
require downloading model files separately.

```bash
python alpha5/tests/experiments/preprocessing_method.py images/ best.pt \
  --sr_method clahe --conf 0.25
```

Note: annotation is drawn on the *processed* image (post-SR), not the original.

---

### `hybrid_pipeline.py` — Hybrid 5-Stage Pipeline

The highest-quality method. Combines full-image and tiled inference with a
smart filtering stage that removes crop detection artifacts.

**Pipeline stages:**

```
Stage 1: Full-image inference   → full_boxes, full_scores, full_classes
Stage 2: Crops inference + WBF  → crops_boxes (fused per-crop detections)
Stage 3: Smart filter crops     → filtered_boxes (remove suspicious fragments)
Stage 4: Merge full + filtered  → merged via WBF
Stage 5: Confidence filter      → class priority (trash deprioritization)
```

**Stage 3 smart filter logic** (the key innovation):

For each crop detection, compute max IoU against same-class full-image detections:

| IoU range | Interpretation | Action |
|---|---|---|
| `>= high_iou` (default 0.85) | Same object seen by both views | **Keep** (validated) |
| `< suspect_iou` (default 0.3) | No overlap — new small object | **Keep** (new find) |
| Between suspect and high | Partial overlap — likely crop fragment | **Discard** |

This logic keeps the best of both worlds: full-image gets large objects,
crops add only genuinely new small objects that the full view missed.

**Stage 5 confidence filter**: groups overlapping detections (IoU > 0.5).
In a group of 2+ high-confidence (> 0.5) detections, picks the highest
confidence one — but skips `trash` (class 7) if a more specific class
is also present at high confidence.

```bash
python alpha5/tests/experiments/hybrid_pipeline.py images/ best.pt \
  --crops 6 --overlap 0.2 \
  --high_iou 0.85 --suspect_iou 0.3 --merge_iou 0.5 \
  --save_intermediate --draw_grid
```

`--save_intermediate` saves images for stages 1, 2, and 3 separately — useful
for debugging the filter behavior. Progress bar shows
`Full=N | Crops=N | Final=N` per image.

---

### `sahi_dir.py` — SAHI Sliced Inference

Wrapper around the [SAHI](https://github.com/obss/sahi) library's
`get_sliced_prediction`. SAHI handles its own crop tiling and NMS internally.

```bash
pip install sahi
python alpha5/tests/experiments/sahi_dir.py images/ best.pt output/ \
  --slice_height 320 --slice_width 320 \
  --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 \
  --conf 0.25 --format jpg
```

Handles both old SAHI (no `export_format`) and new SAHI (with `export_format`)
by trying the newer signature first and falling back gracefully.
Filenames are sanitized to avoid collisions when processing subdirectories.

---

### `patched_inference_alpha5.py` — patched-yolo-infer Wrapper

Uses the [`patched-yolo-infer`](https://github.com/Koldim2001/patched_yolo_infer)
library (`MakeCropsDetectThem` + `CombineDetections`) for patch-based inference.

```bash
pip install patched-yolo-infer
python alpha5/tests/experiments/patched_inference_alpha5.py images/ best.pt \
  --patch_size 640 --overlap 0.25 --nms_threshold 0.25
```

`--save_comparison`: runs standard YOLO on full image alongside, prints
detection count difference (`Patched: N | Baseline: N | Diff: +N`).

---

## Shared Utilities

### `crop_utils.py`

| Component | Description |
|-----------|-------------|
| `UniformCrops` | Computes a sqrt(N) x ceil(N/sqrt(N)) overlapping grid. Strides: `cell * (1 - overlap)`. Clamps coordinates to image bounds. `crops_number` must be even and positive. |
| `draw_crop_grid(img, coords)` | Draws crop rectangles on a copy of the image. Returns the annotated BGR array. |
| `iter_images(source, recursive)` | Lists `.jpg/.jpeg/.png` files from a file or directory, sorted alphabetically. |

**Grid math for 4 crops, overlap=0.2, image width W:**
- Grid: 2 rows x 2 cols
- `cell_w = W / (2 - 1*0.2) = W / 1.8`
- `stride_w = cell_w * 0.8`

---

### `wbf_utils.py`

| Function | Description |
|----------|-------------|
| `compute_iou_xyxy(a, b)` | IoU for two `[x1,y1,x2,y2]` boxes. Returns 0 if union=0. |
| `weighted_boxes_fusion(boxes, scores, classes, iou_thres, skip_box_thr)` | Per-class BFS clustering: boxes within `iou_thres` of each other form a cluster. Fused box = score-weighted average of corners. Fused score = cluster max. Boxes below `skip_box_thr` are dropped before clustering. |
| `greedy_nms_classwise(boxes, scores, classes, iou_thres)` | Per-class greedy NMS sorted by descending confidence. Returns keep-indices. |
| `deduplicate_detections(boxes, scores, classes, iou_threshold, trash_class_id, prioritize_non_trash, keep_all)` | When two boxes overlap above `iou_threshold`: if `prioritize_non_trash=True` and one is `trash` (class 7), the specific class wins; otherwise highest confidence wins. `keep_all=True` bypasses deduplication entirely. |

**WBF vs NMS tradeoff:**
- NMS discards all but the highest-confidence box in a cluster — can miss objects
  that partially overlap at tile boundaries.
- WBF merges overlapping boxes by averaging — smoother box positions, retains
  confidence signal from multiple views.

---

### `crop_maker.py`

Extracts and saves crops from images without running inference. Useful for
inspecting crop coverage before committing to a tile size.

```bash
python alpha5/tests/experiments/crop_maker.py images/ \
  --crops 4 --overlap 0.2 --save_crops --draw_grid
```

Outputs: `<stem>_grid_N.jpg` (annotated with crop indices) and
`<stem>_crops_N/crop_00.jpg ... crop_0N.jpg`.

---

### `pair_concat.py`

Concatenates images from two directories side-by-side by alphabetical order.
Both images are height-matched (shorter one is resized, aspect ratio preserved).

```bash
python alpha5/tests/experiments/pair_concat.py originals/ predictions/ \
  --out_dir comparison/ --suffix _side_by_side
```

Warns if directories have different image counts; processes `min(N_left, N_right)` pairs.

---

### `test_dup.py`

Visual diagnostic tool. Runs inference on a single image and saves **three versions**:

1. `_1_all.jpg` — all raw detections (low threshold recommended, e.g. `--conf 0.15`)
2. `_2_dedup.jpg` — after deduplication (highest confidence wins)
3. `_3_dedup_smart.jpg` — after deduplication with trash deprioritization

Prints per-class distribution and average confidence for version 3.

```bash
python alpha5/tests/experiments/test_dup.py image.jpg best.pt \
  --conf 0.15 --iou_dedup 0.5 --trash_id 7
```

---

## Visualizer GUI

### `tests/visualizer/run_visualizer.py`

Interactive side-by-side comparison of inference methods. Built with Tkinter.

```bash
cd alpha5/tests/visualizer
python run_visualizer.py
```

**Features:**
- Load model (`.pt`) and image via file dialog
- Select any registered inference method from the method registry
- Zoom-to-cursor with mousewheel; click-drag pan; double-click to fit
- Dark theme (`#111118` background)
- Inference runs in a background thread so the UI stays responsive

**Inference method registry** (`inference_methods.py`): methods are registered
by name. The GUI populates its dropdown from `get_available_methods()`.
Add a new method by implementing the interface and registering via `get_method()`.

---

## Docker

```bash
docker build -f alpha5/Dockerfile -t alpha5:latest .
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" alpha5:latest
```

Training output goes to `/ultralytics/USER/runs/detect/train/` inside the
container, which maps to `./runs/detect/train/` on the host via the volume mount.

---

## Dependencies

| Package | Used by |
|---------|---------|
| `ultralytics` | All train/inference/val scripts |
| `torch` | train scripts, CUDA detection |
| `opencv-python` | All inference scripts (image I/O, CLAHE, unsharp mask) |
| `numpy` | All inference scripts (box math, NMS, WBF) |
| `pandas` | `val_yolo.py` (CSV export), `best_epoch.py` |
| `matplotlib` | `val_yolo.py` (bar chart) |
| `Pillow` | `yolo2coco.py` (image size reading), visualizer |
| `tqdm` | All inference scripts (progress bars) |
| `psutil` | `inference.py` (RSS memory measurement) |
| `pyyaml` | Training scripts (hyperparameter YAML) |
| `sahi` | `sahi_dir.py` only |
| `patched-yolo-infer` | `patched_inference_alpha5.py` only |
| `basicsr`, `realesrgan` | `preprocessing_method.py` (`real_esrgan` mode only) |
