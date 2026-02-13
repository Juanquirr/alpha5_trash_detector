# Alpha5 - Trash Detection System

Trash detection system using YOLOv11 with multiple inference strategies and an interactive visualization interface. This repository provides a comprehensive toolkit for object detection with emphasis on waste classification across diverse environmental conditions.

## Author

- Juan Carlos Rodríguez Ramírez

## Project Overview

The objective of this project is to develop a robust object detection model capable of identifying different types of waste in various environments. The system is designed with potential integration into PLOCAN camera streams for automated marine waste detection.

Key features:
- YOLOv11 models trained with Bayesian hyperparameter optimization
- Six distinct inference methods with configurable parameters
- Deduplication system with class-specific prioritization
- Interactive GUI for method comparison and visualization
- Support for tiled inference, multi-scale ensembles, and test-time augmentation

## Dataset

The model is trained on a custom multi-class trash dataset with the following classes:

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
  7: trash
```

Class 7 (trash) serves as a generic fallback category when specific classification is uncertain.

## Training

This repository includes standalone scripts to prepare datasets, train Ultralytics YOLO models, and run hyperparameter tuning.

### Dataset Preparation

#### 1) Instance-stratified split (by object instances)
Use `img_stratifier.py` to split a mixed folder (images and YOLO `.txt` labels in the same directory) into `train/`, `val/`, and `test/` so that the *number of labeled instances per class* is balanced across splits.

```bash
python img_stratifier.py /path/to/mixed_dir \
  --out_dir /path/to/output_dir \
  --ratios 0.7 0.2 0.1 \
  --seed 42
```

The script copies files into the standard YOLO layout: `output_dir/{train,val,test}/images` and `output_dir/{train,val,test}/labels`.

#### 2) Create empty labels for negative samples
Use `create_empty_labels.py` to create empty label files for images that do not have a corresponding `.txt` label file (useful when you keep negative samples).

```bash
# Only report dataset structure (counts images, labels, empty/non-empty, missing labels)
python create_empty_labels.py /path/to/split_dir --check_only

# Dry run (prints what would be created)
python create_empty_labels.py /path/to/split_dir --dry_run

# Create missing empty labels under /path/to/split_dir/labels
python create_empty_labels.py /path/to/split_dir
```

The expected split structure is `/path/to/split_dir/images` and `/path/to/split_dir/labels`.

### Train a YOLO Model

Use `train_yolo.py` to run Ultralytics training with configurable epochs, image size, workers, optimizer, and optional hyperparameter overrides from a YAML file.

```bash
python train_yolo.py /path/to/data.yaml yolo11x.pt \
  --epochs 300 \
  --batch -1 \
  --imgsz 640 \
  --workers 4 \
  --patience 15 \
  --device cuda \
  --project /ultralytics/USER/runs/detect/train \
  --name alpha5_yolo11 \
  --optimizer auto
```

`train_yolo.py` also logs per-epoch `metrics/mAP50(B)` through a training callback and tracks a patience counter based on improvements of mAP50.

#### Optional: override hyperparameters
You can pass a YAML file via `--hyperparams` to override Ultralytics training hyperparameters

```bash
python train_yolo.py /path/to/data.yaml yolo11x.pt \
  --epochs 200 \
  --imgsz 640 \
  --hyperparams /path/to/hparams.yaml
```

If `close_mosaic` is present in that YAML, the script casts it to an integer before calling `model.train()`.

### Hyperparameter Tuning

Use `hyperparam_yolo_tunning.py` to run Ultralytics `model.tune()` with a fixed number of epochs per iteration and a specified number of iterations.

```bash
python hyperparam_yolo_tunning.py /path/to/data.yaml yolo11x.pt 50 20 \
  --batch -1 \
  --imgsz 640 \
  --patience 15 \
  --device cuda \
  --name tune_exp \
  --project /ultralytics/USER/runs/detect/tune
```

You can provide additional initial tuning kwargs through `--tune_kwargs` as a YAML file; the script filters out reserved keys such as `data`, `model`, `epochs`, `iterations`, `batch`, `imgsz`, `patience`, `device`, `name`, `project`, and `resume`.

```bash
python hyperparam_yolo_tunning.py /path/to/data.yaml yolo11x.pt 30 25 \
  --tune_kwargs /path/to/tune_kwargs.yaml
```

If `close_mosaic` is provided in `--tune_kwargs`, the script casts it to an integer before calling `model.tune()`.

## Validation and Inference

```
ALPHA5_val/visualizer/
├── alpha5_base.py          # Base classes (InferenceMethod, InferenceResult)
├── utils.py                # Utilities (crops, WBF, NMS, deduplication)
├── inference_methods.py    # Six inference method implementations
├── alpha5_visualizer.py    # GUI application for interactive comparison
├── run_visualizer.py       # Launcher script for GUI
└── test_methods.py         # CLI tool for batch processing
```

### Inference Methods

1. **Basic**: Standard YOLO inference with configurable confidence and IoU thresholds
2. **Tiled**: Uniform grid cropping with Weighted Boxes Fusion (WBF) or Non-Maximum Suppression (NMS)
3. **MultiScale**: Ensemble inference across multiple image resolutions
4. **TTA**: Test-Time Augmentation with flips and optional brightness variations
5. **SuperRes**: Enhanced preprocessing with CLAHE, Unsharp Mask, or both
6. **Hybrid**: Two-stage pipeline combining full image and cropped detections

## Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Ultralytics YOLO
- OpenCV
- NumPy
- Pillow
- Tkinter (for GUI)

### Docker Environment

Build the Docker image:

```bash
docker build -f ./Dockerfile -t alpha5:latest .
```

Run container with GPU support:

```bash
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/workspace" \
  --name alpha5_container \
  alpha5:latest
```

## Usage of GUI Application

Launch the interactive visualizer:

```bash
python run_visualizer.py
```

Features:
- Load YOLO model (.pt files)
- Load test images
- Configure method-specific parameters via GUI
- Execute multiple methods simultaneously
- Compare results side-by-side
- Export annotated images
- Adjustable confidence and IoU thresholds with real-time updates

### Command-Line Interface

Run inference methods from terminal:

```bash
python test_methods.py image.jpg model.pt --output results/
```

The script will run all available methods and save results to the output directory.

### Programmatic Usage

```python
from inference_methods import get_method
from ultralytics import YOLO
import cv2

# Load model and image
model = YOLO('yolov11x.pt')
image = cv2.imread('test_image.jpg')

# Get inference method
tiled = get_method('tiled')

# Configure parameters
params = {
    'conf': 0.25,
    'iou': 0.5,
    'crops': 6,
    'overlap': 0.2,
    'fusion': 'wbf',
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}

# Run inference
result = tiled.run(image, model, params)

# Access results
print(f"Detections: {result.num_detections}")
print(f"Elapsed time: {result.elapsed_time:.2f}s")
cv2.imwrite('output.jpg', result.image)
```

## Deduplication System

The deduplication module eliminates redundant detections while optionally prioritizing specific classes over generic ones.

### Parameters

- `deduplicate`: Enable/disable deduplication (bool)
- `dedup_iou`: IoU threshold for considering detections as duplicates (float, 0.0-1.0)
- `trash_class_id`: ID of the generic trash class (int, default: 7)
- `prioritize_specific`: Prefer specific classes over generic trash (bool)

### Logic

When `prioritize_specific=True`:
- If two detections overlap (IoU >= `dedup_iou`):
  - Specific class (plastic, metal, etc.) takes precedence over generic trash
  - Among detections of equal priority, highest confidence wins

When `prioritize_specific=False`:
- Highest confidence detection always wins regardless of class

### Method-Specific Defaults

| Method      | deduplicate | Rationale                          |
|-------------|-------------|-------------------------------------|
| Basic       | False       | Single-pass inference, minimal overlap |
| Tiled       | True        | Crops generate overlapping detections |
| MultiScale  | True        | Multiple resolutions cause duplicates |
| TTA         | True        | Augmentations create redundancy |
| SuperRes    | False       | Single preprocessing step |
| Hybrid      | True        | Full + crops strategy requires merging |

## Method Parameters

### Basic

```python
{
    'conf': 0.25,           # Confidence threshold
    'iou': 0.45,            # NMS IoU threshold
    'imgsz': 640,           # Input image size
    'deduplicate': False,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

### Tiled

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'crops': 4,              # Number of crops (must be even)
    'overlap': 0.2,          # Overlap ratio between crops
    'fusion': 'wbf',         # Fusion method: 'wbf' or 'nms'
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

### MultiScale

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'scales': [640, 960, 1280],  # Image sizes for multi-scale inference
    'nms_thresh': 0.5,           # Final NMS threshold
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

### TTA (Enhanced)

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'tta_iou': 0.5,          # IoU threshold for TTA fusion
    'imgsz': 640,
    'use_flips': True,       # Enable horizontal/vertical flips
    'use_brightness': False, # Enable brightness augmentation (optional)
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

**TTA Augmentations:**
- Original image
- Horizontal flip
- Vertical flip
- Combined (horizontal + vertical) flip
- Brighter version (if `use_brightness=True`)
- Darker version (if `use_brightness=True`)

### SuperRes (Enhanced)

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'imgsz': 640,
    'sr_method': 'clahe',    # Preprocessing: 'clahe', 'unsharp', or 'both'
    'clahe_clip': 3.0,       # CLAHE contrast limit
    'clahe_tile': 8,         # CLAHE tile grid size (8x8)
    'unsharp_sigma': 1.0,    # Gaussian blur sigma for unsharp mask
    'unsharp_strength': 1.5, # Sharpening intensity
    'deduplicate': False,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

**SuperRes Methods:**
- `'clahe'`: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space L channel
- `'unsharp'`: Apply Unsharp Mask sharpening to enhance edges
- `'both'`: Apply CLAHE first, then Unsharp Mask for maximum enhancement

### Hybrid

```python
{
    'conf': 0.25,
    'crops': 6,
    'overlap': 0.2,
    'merge_iou': 0.5,        # IoU for merging full + crops detections
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

## Performance Considerations

### Method Selection Guide

- **Basic**: Fast inference on standard images, minimal overhead
- **Tiled**: Recommended for images with small objects or high resolution
- **MultiScale**: Robust detection across object sizes, higher computational cost
- **TTA**: Improved accuracy through augmentation, 4-6x inference time (depending on augmentations)
- **SuperRes**: Benefits low-quality or low-contrast images
- **Hybrid**: Maximum detection quality, highest computational cost

### Computational Requirements

Tested on YOLOv11x with input resolution 640x640:

| Method      | Relative Speed | Memory Overhead | Use Case                  |
|-------------|----------------|-----------------|---------------------------|
| Basic       | 1.0x           | Low             | Standard images           |
| Tiled       | 1.5-2.5x       | Medium          | High-res or small objects |
| MultiScale  | 3.0-4.0x       | Medium-High     | Variable object sizes     |
| TTA         | 4.0-6.0x       | Low-Medium      | Maximum accuracy needed   |
| SuperRes    | 1.1-1.3x       | Low             | Poor image quality        |
| Hybrid      | 2.5-3.5x       | Medium-High     | Critical applications     |

*Note: TTA speed varies based on enabled augmentations (flips only: ~4x, with brightness: ~6x)*

## Model Information

This project uses YOLOv11x for optimal performance on small and complex object detection. Alternative model sizes can be substituted:

- `yolov11n.pt`: Nano (fastest, lowest accuracy)
- `yolov11s.pt`: Small
- `yolov11m.pt`: Medium (recommended for limited hardware)
- `yolov11l.pt`: Large
- `yolov11x.pt`: Extra-large (highest accuracy, used in this project)

For resource-constrained environments, use `yolov11m.pt` with `imgsz=416`.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alpha5,
  author = {Rodríguez Ramírez, Juan Carlos},
  title = {Alpha5: Trash Detection System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Juanquirr/alpha5}}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Ultralytics YOLO for the object detection framework
- PLOCAN for project support and marine environment integration goals
- Kolesnikov Dmitry for patched inference implementation contributions
