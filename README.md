# Alpha5 — Trash Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv26](https://img.shields.io/badge/Model-YOLOv26-purple)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://www.docker.com/)
[![LICENSE](https://img.shields.io/badge/License-MIT-green)](LICENSE.md)

**Alpha5** is a robust waste detection system built on the YOLO26 architecture,
featuring six configurable inference strategies and an interactive GUI for visual comparison of results.

</div>

---

## Overview

Alpha5 is designed to detect and classify urban and marine waste across diverse
environmental conditions. The system is oriented towards integration with real-world
camera streams — including PLOCAN marine monitoring infrastructure — and provides a
complete toolkit covering dataset preparation, model training, hyperparameter tuning,
multi-strategy inference, and interactive result visualization.

The project targets eight waste categories, combining fine-grained classes with a
generic fallback to handle uncertain predictions:

```yaml
nc: 8
names:
  0: plastic_bottle
  1: glass
  2: can
  3: plastic_bag
  4: metal_scrap
  5: plastic_wrapper
  6: trash_pile
  7: trash          # Generic fallback class
```

---

## Key Features

- **YOLO26** backbone trained with hyperparameter tuning.
- **Six inference strategies** covering from real-time single-pass to high-accuracy
  hybrid approaches
- **Deduplication system** with class-specific prioritization (specific classes over
  generic `trash`)
- **Interactive GUI** for side-by-side method comparison with zoom, pan and
  collapsible panels
- **CLI and programmatic API** for batch processing and integration
- **Full Docker support** with GPU passthrough

Note: The code supports any **Ultralytics** YOLO model.  

---

## Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Ultralytics YOLO
- OpenCV
- NumPy
- Pillow
- Tkinter (for GUI)

### Docker (Recommended)

Build the image:

```bash
docker build -f ./Dockerfile -t alpha5:latest .
```

Run with GPU support:

```bash
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" \
  --name alpha5_container \
  alpha5:latest
```

---

## Training Pipeline

The training workflow consists of three independent, composable stages: dataset
preparation, model training, and hyperparameter optimization.

### 1. Dataset Preparation

#### Instance-stratified split

`img_stratifier.py` splits a flat folder of images and YOLO `.txt` labels into
`train/`, `val/`, and `test/` subsets, balancing the **number of annotated instances
per class** across splits — not just image counts.

```bash
python img_stratifier.py /path/to/mixed_dir \
  --out_dir /path/to/output_dir \
  --ratios 0.7 0.2 0.1 \
  --seed 42
```

Output follows the standard YOLO layout:
`output_dir/{train,val,test}/{images,labels}`.

#### Negative sample labels

`create_empty_labels.py` creates empty `.txt` label files for images without
annotations, which is required when negative samples are included in training.

```bash
# Inspect dataset structure only
python create_empty_labels.py /path/to/split_dir --check_only

# Dry run (preview without writing)
python create_empty_labels.py /path/to/split_dir --dry_run

# Write missing empty label files
python create_empty_labels.py /path/to/split_dir
```

---

### 2. Model Training

`train_yolo.py` wraps the Ultralytics `model.train()` API with additional features:
a per-epoch `metrics/mAP50(B)` logging callback and a manual patience counter based
on mAP50 improvement.

```bash
python train_yolo.py /path/to/data.yaml yolo26x.pt \
  --epochs 300 \
  --batch -1 \
  --imgsz 640 \
  --workers 4 \
  --patience 15 \
  --device cuda \
  --project /ultralytics/USER/runs/detect/train \
  --name alpha5_yolo26 \
  --optimizer auto
```

Hyperparameters can be overridden at runtime via a YAML file:

```bash
python train_yolo.py /path/to/data.yaml yolo26x.pt \
  --epochs 200 \
  --imgsz 640 \
  --hyperparams /path/to/hparams.yaml
```

> **Note:** If `close_mosaic` is present in the hyperparameter file, the script
> automatically casts it to `int` before passing it to Ultralytics.

---

### 3. Hyperparameter Tuning

`hyperparam_yolo_tunning.py` runs Ultralytics `model.tune()` (Bayesian optimization)
for a configurable number of iterations, each trained for a fixed number of epochs.

```bash
python hyperparam_yolo_tunning.py /path/to/data.yaml yolo26x.pt 50 20 \
  --batch -1 \
  --imgsz 640 \
  --patience 15 \
  --device cuda \
  --name tune_exp \
  --project /ultralytics/USER/runs/detect/tune
```

Additional tuning kwargs can be provided via `--tune_kwargs` as a YAML file. Reserved
keys (`data`, `model`, `epochs`, `iterations`, `batch`, `imgsz`, `patience`,
`device`, `name`, `project`, `resume`) are automatically filtered out to prevent
conflicts.

---

## Inference Methods

All methods are implemented in `inference_methods.py` and share a common interface
through the `InferenceMethod` base class. Each method returns an `InferenceResult`
object containing bounding boxes, scores, class labels, elapsed time, and an
annotated image.

### 1. Basic

Standard single-pass YOLO inference. Applies confidence filtering and NMS, with
optional deduplication. This is the reference baseline for all other methods.

**Best for:** Real-time applications, large/medium objects, limited compute.

---

### 2. Tiled

Divides the image into an overlapping grid of crops (default: 4 crops, 20% overlap),
runs inference on each crop independently, transforms detections back to global
coordinates, and merges them using **Weighted Boxes Fusion (WBF)** or **NMS**.

The overlap strategy ensures that objects near crop boundaries are detected by at
least one crop, and WBF consolidates duplicate detections from adjacent tiles into a
single confident prediction.

**Best for:** High-resolution images, dense small-object scenes.

---

### 3. MultiScale

Runs inference at multiple resolutions (default: `[640, 960, 1280]`) and merges
all predictions using cross-scale NMS. Each scale is optimized for a different
object size range; objects detected at multiple scales receive implicitly higher
reliability.

**Best for:** Scenes with highly variable object sizes (near/far objects, drone
footage).

---

### 4. Test-Time Augmentation (TTA)

Generates augmented versions of the input (horizontal flip, vertical flip, combined
flip, optional brightness ±20%), runs inference on each, reverses the geometric
transforms, and pools all detections using TTA-specific NMS with a voting mechanism
for class labels.

**Best for:** Objects with variable orientations, symmetric shapes (bottles, cans),
challenging lighting.

---

### 5. SuperResolution Preprocessing

Applies image enhancement before standard single-pass inference. Supports three
modes:

| Mode | Technique | Effect |
|------|-----------|--------|
| `clahe` | Contrast Limited Adaptive Histogram Equalization on the L channel (LAB space) | Reveals objects in shadows and low-contrast areas |
| `unsharp` | Gaussian blur subtraction with configurable strength | Sharpens edges and object boundaries |
| `both` | Sequential CLAHE → Unsharp Mask | Maximum feature clarity |

**Best for:** Compressed, underexposed, or low-contrast imagery.

---

### 6. Hybrid

Combines a full-image inference pass with a tiled inference pass (default: 6 crops)
and merges both streams using **WBF**. The full-image stream captures large objects
with global context; the crop stream captures small objects at higher effective
resolution. WBF weights detections by confidence and produces coordinate-averaged
boxes, boosting confidence for objects detected in both streams.

**Best for:** Critical applications requiring maximum recall across all object sizes.

---

### Performance Reference

Benchmarked on YOLOv11x at 640×640 input resolution:

| Method | Relative Speed | Small Objects | Large Objects | Recommended Use Case |
|---|---|---|---|---|
| Basic | 1.0× | ⭐⭐ Fair | ⭐⭐⭐⭐⭐ Excellent | Real-time, baseline |
| Tiled | 1.5–2.5× | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Good | High-res, dense scenes |
| MultiScale | 3.0–4.0× | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good | Variable object sizes |
| TTA | 4.0–6.0× | ⭐⭐ Fair | ⭐⭐⭐⭐ Good | Orientation/lighting variability |
| SuperRes | 1.1–1.3× | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | Low-quality imagery |
| **Hybrid** | **2.5–3.5×** | **⭐⭐⭐⭐⭐ Excellent** | **⭐⭐⭐⭐⭐ Excellent** | **Research, validation** |

---

## Deduplication System

All methods support an optional post-inference deduplication step implemented in
`utils.py`. When `prioritize_specific=True`, overlapping detections are resolved by
keeping the most specific class over the generic `trash` class (ID 7). Among
detections of equal priority, the highest-confidence detection wins.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `deduplicate` | bool | method-dependent | Enable deduplication |
| `dedup_iou` | float | 0.5 | IoU threshold to consider two detections as duplicates |
| `trash_class_id` | int | 7 | ID of the generic fallback class |
| `prioritize_specific` | bool | True | Specific classes take precedence over generic trash |

Methods that generate redundancy (Tiled, MultiScale, TTA, Hybrid) have
`deduplicate=True` by default; methods based on a single inference pass (Basic,
SuperRes) default to `False`.

---

## Usage

### GUI Application

Launch the interactive visualizer:

```bash
python run_visualizer.py
```

**Features:**
- Load any `.pt` YOLO model
- Load test images and configure method-specific parameters via GUI (⚙️ buttons)
- Execute multiple methods simultaneously and compare side-by-side
- Double-click zoom with pan and collapsible panels (v4+)
- Adjustable confidence and IoU thresholds with real-time updates
- Export annotated images

### Command-Line Interface

Run all inference methods on a single image and save results:

```bash
python test_methods.py image.jpg model.pt --output results/
```

### Programmatic API

```python
from inference_methods import get_method
from ultralytics import YOLO
import cv2

model = YOLO('yolo26x.pt')
image = cv2.imread('test_image.jpg')

hybrid = get_method('hybrid')

params = {
    'conf': 0.25,
    'crops': 6,
    'overlap': 0.25,
    'merge_iou': 0.55,
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}

result = hybrid.run(image, model, params)

print(f"Detections: {result.num_detections}")
print(f"Elapsed time: {result.elapsed_time:.2f}s")

cv2.imwrite('output.jpg', result.image)
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{alpha5,
  author       = {Rodríguez Ramírez, Juan Carlos},
  title        = {Alpha5: Trash Detection System},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Juanquirr/alpha5_trash_detector}}
}
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- [PLOCAN](https://www.plocan.eu/) for project support and marine environment
  integration goals
- Kolesnikov Dmitry for patched inference implementation contributions

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE.md) for
details.
