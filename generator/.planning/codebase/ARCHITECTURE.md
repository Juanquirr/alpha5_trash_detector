# ARCHITECTURE.md — System Architecture

## Pattern

**Pipeline-based CLI tool** — No server/API in active use. A CLI runner (`run.py`) drives a sequential image processing pipeline that:
1. Loads source images from disk
2. Detects water regions
3. Places trash objects at valid water positions
4. Inpaints each object using a FLUX model
5. Saves annotated outputs (PNG + YOLO labels)

## High-Level Architecture

```
run.py (CLI entry point)
    │
    ├── cmd_fill()   → Generates full dataset
    └── cmd_test()   → Compares models on subset
            │
            ▼
    core/pipeline.py (orchestrator)
        ├── load_model()         → Model factory (lazy import)
        ├── process_image()      → Full per-image pipeline
        ├── insert_object()      → Single-object placement + inpaint
        └── run_inpaint()        → Model dispatch layer
            │
            ├─── Water detection (pluggable)
            │    core/water_detector.py (shared utils + HSV default)
            │    core/water_detector_hsv.py
            │    core/water_detector_otsu.py
            │    core/water_detector_kmeans.py
            │    core/water_detector_flood.py
            │    core/water_detector_sam.py
            │
            └─── Inpainters (pluggable, ABC pattern)
                 core/inpainters/base.py (ImageInpainter ABC)
                 core/inpainters/flux_fill.py    (text-conditioned)
                 core/inpainters/flux_canny.py   (edge-guided)
                 core/inpainters/flux_redux.py   (visual reference)
                 core/inpainters/flux_kontext.py (in-context edit)
```

## Layers

| Layer | Files | Responsibility |
|-------|-------|---------------|
| CLI | `run.py` | Argument parsing, subcommands, loop control, CSV logging |
| Orchestration | `core/pipeline.py` | Full pipeline coordination, model dispatch, config |
| Water Detection | `core/water_detector*.py` | Water mask generation (5 interchangeable methods) |
| Inpainting | `core/inpainters/` | FLUX model wrappers (4 strategies) |
| Image Utilities | `core/image_utils.py` | Resize, mask creation, crop region, debug overlay, YOLO bbox |
| Config | `core/constants.py`, `config/prompts.csv` | Pipeline parameters and class prompts |

## Key Data Flow

```
Input image (JPG/PNG)
    → prepare_image()       resize to ≤1024px, round to multiples of 16
    → create_water_mask()   binary mask, 255=water
    → find_water_positions() N random (cx,cy,class_id,w,h) tuples within water
    → for each position:
        → load prompt from prompts_by_class[class_id]
        → insert_object()
            → compute_crop_region() (if use_crop=True)
            → create_mask()         soft elliptical inpaint mask
            → run_inpaint()         → model.inpaint(image, mask, prompt)
            → paste crop back (if crop mode)
            → compute_yolo_bbox()
    → save PNG, .txt (YOLO), _debug.png, _water_mask.png
```

## Abstraction Points

### Water Detector (pluggable)
Selected via `--water-method` CLI flag. All detectors expose:
```python
def create_water_mask(image_np: np.ndarray) -> np.ndarray  # returns binary mask
```
Methods: `hsv` (default), `otsu`, `kmeans`, `flood`, `sam`
Loaded dynamically via `importlib` in `core/pipeline.py`.

### Inpainter (pluggable, ABC)
`core/inpainters/base.py` defines `ImageInpainter` ABC:
```python
def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, **kwargs) -> Image.Image
```
`FluxKontextInpainter` also adds `compute_bbox()` for diff-based bbox detection.
`FluxReduxInpainter` adds `class_id` kwarg for reference image selection.

## Crop vs Full-Image Mode

**Default: crop mode (`use_crop=True`)**
- Extracts a local crop (320–640px) around the target position
- Inpaints the crop (model sees focused water context)
- Pastes result back at original position
- Better integration quality

**Legacy: full-image mode (`--no-crop`)**
- Inpaints the entire image with a small mask
- Faster but lower contextual quality

## YOLO Annotation Format

All outputs use YOLO format:
```
{class_id} {x_center} {y_center} {width} {height}
```
Coordinates normalized to [0,1] relative to full image dimensions.
One `.txt` file per image, one line per inserted object.

## ProcessConfig

Central configuration dataclass in `core/pipeline.py`:
```python
@dataclass
class ProcessConfig:
    n_objects: int | None    # None = random(2,3)
    use_crop: bool = True
    output_suffix: str       # "_synth" (fill) or "_result" (test)
    min_objects: int = 2
    max_objects: int = 3
    log_fields: list
    water_method: str = "hsv"
```
