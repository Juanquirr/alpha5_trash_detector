# Alpha5 - Advanced Trash Detection System

Trash detection system using YOLOv11 with multiple inference strategies and an interactive visualization interface. This repository provides a comprehensive toolkit for object detection with emphasis on waste classification across diverse environmental conditions.

## Author

Juan Carlos Rodríguez Ramírez

## Project overview

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

## Architecture

### Core components

```
alpha5/
├── alpha5_base.py          # Base classes (InferenceMethod, InferenceResult)
├── utils.py                # Utilities (crops, WBF, NMS, deduplication)
├── inference_methods.py    # Six inference method implementations
├── alpha5_visualizer.py    # GUI application for interactive comparison
├── run_visualizer.py       # Launcher script for GUI
└── test_methods.py         # CLI tool for batch processing
```

### Inference methods

1. **Basic**: Standard YOLO inference with configurable confidence and IoU thresholds
2. **Tiled**: Uniform grid cropping with Weighted Boxes Fusion (WBF) or Non-Maximum Suppression (NMS)
3. **MultiScale**: Ensemble inference across multiple image resolutions
4. **TTA**: Test-Time Augmentation with horizontal, vertical, and combined flips
5. **SuperRes**: Preprocessing with CLAHE or unsharp masking before inference
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

### Docker environment

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

## Usage

### GUI Application

Launch the interactive visualizer:

```bash
python run_visualizer.py
```

Features:
- Load YOLO model (.pt files)
- Load test images
- Configure method-specific parameters
- Execute multiple methods simultaneously
- Compare results side-by-side
- Export annotated images
- Adjustable confidence and IoU thresholds with real-time updates

### Command-Line interface

Run inference methods from terminal:

```bash
python test_methods.py image.jpg model.pt \
  --methods basic tiled tta \
  --output results/ \
  --conf 0.25 \
  --iou 0.45
```

### Programmatic usage

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

## Deduplication system

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

### Method-specific defaults

| Method      | deduplicate | Rationale                          |
|-------------|-------------|------------------------------------|
| Basic       | False       | Single-pass inference, minimal overlap |
| Tiled       | True        | Crops generate overlapping detections |
| MultiScale  | True        | Multiple resolutions cause duplicates |
| TTA         | True        | Augmentations create redundancy |
| SuperRes    | False       | Single preprocessing step |
| Hybrid      | True        | Full + crops strategy requires merging |

## Method parameters

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
    'scales': ,  # Image sizes for multi-scale inference
    'nms_thresh': 0.5,            # Final NMS threshold
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

### TTA

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'tta_iou': 0.5,          # IoU threshold for TTA fusion
    'imgsz': 640,
    'deduplicate': True,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

### SuperRes

```python
{
    'conf': 0.25,
    'iou': 0.5,
    'imgsz': 640,
    'sr_method': 'clahe',    # Preprocessing: 'clahe' or 'unsharp'
    'deduplicate': False,
    'dedup_iou': 0.5,
    'trash_class_id': 7,
    'prioritize_specific': True
}
```

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

## Performance considerations

### Method selection guide

- **Basic**: Fast inference on standard images, minimal overhead
- **Tiled**: Recommended for images with small objects or high resolution
- **MultiScale**: Robust detection across object sizes, higher computational cost
- **TTA**: Improved accuracy through augmentation, 4x inference time
- **SuperRes**: Benefits low-quality or low-contrast images
- **Hybrid**: Maximum detection quality, highest computational cost

### Computational requirements

Tested on YOLOv11x with input resolution 640x640:

| Method      | Relative Speed | Memory Overhead | Use Case                  |
|-------------|----------------|-----------------|---------------------------|
| Basic       | 1.0x           | Low             | Standard images           |
| Tiled       | 1.5-2.5x       | Medium          | High-res or small objects |
| MultiScale  | 3.0-4.0x       | Medium-High     | Variable object sizes     |
| TTA         | 4.0x           | Low-Medium      | Maximum accuracy needed   |
| SuperRes    | 1.2x           | Low             | Poor image quality        |
| Hybrid      | 2.5-3.5x       | Medium-High     | Critical applications     |

## Model information

This project uses YOLOv11x for optimal performance on small and complex object detection. Alternative model sizes can be substituted:

- `yolov11n.pt`: Nano (fastest, lowest accuracy)
- `yolov11s.pt`: Small
- `yolov11m.pt`: Medium (recommended for limited hardware)
- `yolov11l.pt`: Large
- `yolov11x.pt`: Extra-large (highest accuracy, used in this project)

For resource-constrained environments, use `yolov11m.pt` with `imgsz=416`.

## Training scripts

Training utilities are located in the `experiments/` directory:

- `train_yolo.py`: Main training pipeline with early stopping
- `hyperparam_yolo_tunning.py`: Bayesian hyperparameter optimization
- `img_stratifier.py`: Instance-balanced dataset splitting
- `val_yolo.py`: Comprehensive validation with metrics export

Refer to the original README in `experiments/` for detailed training instructions.

## Citation

If you use this code in your research, please cite:

```
@misc{alpha5,
  author = {Rodríguez Ramírez, Juan Carlos},
  title = {Alpha5: Advanced Trash Detection System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/alpha5}}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Ultralytics YOLO for the object detection framework
- PLOCAN for project support and marine environment integration goals
- Kolesnikov Dmitry for patched inference implementation contributions
