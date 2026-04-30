# CONVENTIONS.md — Code Conventions

## Code Style

- **Language:** Python 3.10+ with modern syntax (`int | None` unions, walrus operator usage possible)
- **Type hints:** Used consistently in function signatures throughout `core/`
- **Docstrings:** Module-level docstrings explain purpose and strategy; function docstrings explain parameters/returns
- **Line length:** Not strictly enforced, ~100 chars typical
- **Imports:** Standard → third-party → local, separated by blank lines. Lazy imports used in `load_model()` to avoid loading unused GPU models

## Naming Patterns

```python
# Constants: UPPER_SNAKE_CASE in core/constants.py
MAX_SIDE = 1024
OBJECT_SIZES = {0: (50, 100, 25, 50), ...}

# Classes: PascalCase
class FluxLocalImageInpainter(ImageInpainter, BaseModel): ...
class ProcessConfig: ...

# Functions: snake_case, verb-first
def process_image(...) -> list[str]: ...
def find_water_positions(...) -> list: ...
def create_water_mask(image_np: np.ndarray) -> np.ndarray: ...

# Private helpers: underscore prefix
def _get_water_detector(method: str): ...
def _find_reference(self, class_id: int) -> Image.Image | None: ...
def _make_canny_control(...) -> Image.Image: ...
```

## Class Patterns

### Inpainter Pattern (Pydantic + ABC)

All inpainters use Pydantic `BaseModel` for config fields and lazy model initialization in `model_post_init`:

```python
class FluxLocalImageInpainter(ImageInpainter, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    _pipe: FluxFillPipeline = None  # Private, initialized in post_init

    def model_post_init(self, __context):
        self._pipe = FluxFillPipeline.from_pretrained(...).to("cuda")

    def inpaint(self, image, mask, prompt, ...) -> Image.Image:
        return self._pipe(...).images[0]
```

### Water Detector Pattern (module-level function)

All water detectors expose a single function at module level:
```python
def create_water_mask(image_np: np.ndarray) -> np.ndarray:
    """Returns binary mask (H,W), dtype=uint8, 255=water."""
    ...
```
Shared post-processing (morphological cleanup, small region removal) is imported from `core/water_detector.py`.

### Config Dataclass Pattern

Pipeline config uses `@dataclass`:
```python
@dataclass
class ProcessConfig:
    n_objects: int | None = None
    use_crop: bool = True
    water_method: str = "hsv"
    log_fields: list = field(default_factory=list)
```

## Error Handling

- **Silent skip pattern:** If water coverage < 1% or no valid positions found, `process_image()` returns `[]` with a printed warning — does not raise
- **None returns:** Functions that may fail to compute a bbox return `None`; callers check and skip
- **No try/except:** Core pipeline code does not use exception handling — errors propagate to CLI
- **Validation:** `load_model()` and `_get_water_detector()` raise `ValueError` for unknown names

## Output / Logging

- **Print-based progress:** `print()` statements throughout `process_image()` for real-time CLI feedback
- **CSV logging:** `csv.DictWriter` for per-image generation logs (append mode, header written on creation)
- **Console separator lines:** `─` (thin) and `═` (thick) ASCII lines separate images/models visually

## Lazy Imports

Heavy FLUX models are imported lazily inside `load_model()` to avoid CUDA memory allocation for unused models:
```python
def load_model(model_name: str, ...):
    if model_name == "fill":
        from core.inpainters.flux_fill import FluxLocalImageInpainter
        return FluxLocalImageInpainter()
```

## Precision

All FLUX models use `torch.bfloat16` for memory efficiency on GPU.

## Image Coordinate System

- PIL: `(width, height)` — used for `image.size`
- NumPy: `(height, width)` — used for array indexing `mask_np[y, x]`
- YOLO: normalized `[0,1]` relative to full image dimensions

## No Tests in Core

The codebase has `pytest` as a dependency but no test files were found. Testing appears to be done manually via `run.py test` subcommand.
