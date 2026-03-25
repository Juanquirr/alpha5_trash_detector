<!-- GSD:project-start source:PROJECT.md -->
## Project

**Marine Trash Synthetic Dataset Generator**

A synthetic dataset generation pipeline that inserts photorealistic floating trash into real harbour and coastal camera images using FLUX diffusion models. It produces YOLO-format annotated training data to close the domain gap between generic trash datasets and the specific marine/coastal environments where a detection model will be deployed.

**Core Value:** Generate synthetic images that are realistic enough to measurably improve a YOLO detection model's real-world performance on harbour camera feeds — not just training metrics.

### Constraints

- **GPU**: Single RTX 5000 Ada 32 GB — limits to one FLUX model loaded at a time; Redux loads two (Redux + Fill) which is tight on VRAM
- **Target environment**: Real harbour/coastal camera images — the generated data must match these specific perspectives and lighting conditions
- **Model architecture**: YOLO-based detection — output must be YOLO-format annotations (class_id, normalised xywh)
- **Reproducibility**: `transformers` is pinned to git HEAD (non-reproducible builds) — needs fixing but is a known constraint for SAM3 compatibility
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Language & Runtime
| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ (inferred from `int | None` type union syntax) |
| Runtime | CPU + CUDA GPU | PyTorch 2.6.0, CUDA 12.4, cuDNN 9 (from Dockerfile) |
| Container | Docker | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` base image |
## Core Frameworks & Libraries
### AI / ML
| Library | Version | Purpose |
|---------|---------|---------|
| `diffusers` | >=0.30.0 | FLUX pipeline orchestration (Fill, Canny, Redux, Kontext) |
| `torch` | 2.6.0 (from base image) | GPU tensor ops, model execution |
| `transformers` | git HEAD (huggingface) | Underlying model support for FLUX pipelines |
| `accelerate` | >=0.33.0 | Multi-device acceleration for diffusers |
| `huggingface_hub` | >=0.24.0 | Model weight downloads from HuggingFace Hub |
| `attention_map_diffusers` | latest | Attention map visualization |
| `torchmetrics` | ~1.7.1 | ML quality metrics (LPIPS etc.) |
| `lpips` | latest | Perceptual image similarity metric |
### Computer Vision
| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ~4.11.0.86 | Canny edge detection, morphological ops, color space |
| `pillow` | ~11.2.1 | Image I/O, resize, mask creation, compositing |
| `numpy` | ~2.0.2 | Array ops throughout pipeline |
| `scipy` | ~1.13.1 | Scientific computing utilities |
### Data / Object Detection
| Library | Version | Purpose |
|---------|---------|---------|
| `autodistill` | ~0.1.29 | Auto-labeling framework |
| `autodistill-grounded-sam` | latest | Grounded SAM for water detection (SAM method) |
| `scikit-learn` | latest | ML utilities |
| `roboflow` | latest | Dataset management / upload |
### Web / API
| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ~0.115.12 | Web API framework (present but not currently used in CLI) |
| `uvicorn` | latest | ASGI server for FastAPI |
| `pydantic` | ~2.11.3 | Data validation, used as model base class for inpainters |
| `requests` | ~2.32.3 | HTTP client |
| `websocket-client` | ~1.8.0 | WebSocket support |
| `tornado` | ~6.4.2 | Async networking |
### Utilities
| Library | Version | Purpose |
|---------|---------|---------|
| `openai` | latest | OpenAI API client (present, not used in core pipeline) |
| `sentencepiece` | latest | Tokenization for transformers |
| `python-dotenv` | ~1.1.0 | Environment variable loading |
| `questionary` | ~2.0.1 | CLI interactive prompts |
| `rich` | latest | Rich terminal output |
| `pyfiglet` | latest | ASCII art banners |
| `matplotlib` | ~3.9.4 | Plotting / visualization |
| `seaborn` | latest | Statistical data visualization |
| `pandas` | ~2.2.3 | Tabular data handling |
| `pytest` | latest | Test framework |
## HuggingFace Models Used
| Model ID | Inpainter | Purpose |
|----------|-----------|---------|
| `black-forest-labs/FLUX.1-Fill-dev` | FluxLocalImageInpainter, FluxReduxInpainter | Text-conditioned inpainting (baseline) |
| `black-forest-labs/FLUX.1-Canny-dev` | FluxCannyInpainter | Edge-guided image generation |
| `black-forest-labs/FLUX.1-Redux-dev` | FluxReduxInpainter | Visual embedding extraction from reference images |
| `black-forest-labs/FLUX.1-Kontext-dev` | FluxKontextInpainter | In-context text-based image editing |
## Configuration
| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependency pinning |
| `Dockerfile` | Container build (GPU runtime) |
| `config/prompts.csv` | Text prompts per trash class (class_id, class_name, prompt) |
| `core/constants.py` | Pipeline constants (sizes, margins, crop params) |
## CLI Entry Point
- `fill` — full dataset generation with FLUX Fill
- `test` — model comparison mode (canny/redux/kontext/all)
## Docker Details
- **Base:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **System deps:** `libgl1-mesa-glx`, `libglib2.0-0`, `git`
- **Port:** 8001 (FastAPI, not currently activated in CMD)
- **CMD:** `/bin/bash` (interactive shell — intended for manual invocation)
- **PYTHONUNBUFFERED:** `1`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Code Style
- **Language:** Python 3.10+ with modern syntax (`int | None` unions, walrus operator usage possible)
- **Type hints:** Used consistently in function signatures throughout `core/`
- **Docstrings:** Module-level docstrings explain purpose and strategy; function docstrings explain parameters/returns
- **Line length:** Not strictly enforced, ~100 chars typical
- **Imports:** Standard → third-party → local, separated by blank lines. Lazy imports used in `load_model()` to avoid loading unused GPU models
## Naming Patterns
## Class Patterns
### Inpainter Pattern (Pydantic + ABC)
### Water Detector Pattern (module-level function)
### Config Dataclass Pattern
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
## Precision
## Image Coordinate System
- PIL: `(width, height)` — used for `image.size`
- NumPy: `(height, width)` — used for array indexing `mask_np[y, x]`
- YOLO: normalized `[0,1]` relative to full image dimensions
## No Tests in Core
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern
## High-Level Architecture
```
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
```
## Abstraction Points
### Water Detector (pluggable)
```python
```
### Inpainter (pluggable, ABC)
```python
```
## Crop vs Full-Image Mode
- Extracts a local crop (320–640px) around the target position
- Inpaints the crop (model sees focused water context)
- Pastes result back at original position
- Better integration quality
- Inpaints the entire image with a small mask
- Faster but lower contextual quality
## YOLO Annotation Format
```
```
## ProcessConfig
```python
```
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
