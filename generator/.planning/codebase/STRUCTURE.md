# STRUCTURE.md — Directory Structure

## Top-Level Layout

```
trash_generator/
├── run.py                      # CLI entry point (fill / test subcommands)
├── water_masks.py              # Standalone water mask utility script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # GPU container (pytorch base)
│
├── core/                       # Core pipeline library
│   ├── __init__.py
│   ├── pipeline.py             # Main orchestrator: load_model, process_image
│   ├── constants.py            # Shared numeric constants
│   ├── prompts.py              # CSV prompt loader
│   ├── image_utils.py          # Image ops: resize, mask, crop, bbox, debug
│   ├── water_detector.py       # Shared utils + HSV re-export + find_water_positions
│   ├── water_detector_hsv.py   # HSV-based water detection (default)
│   ├── water_detector_otsu.py  # Otsu threshold method
│   ├── water_detector_kmeans.py# K-means clustering method
│   ├── water_detector_flood.py # Flood fill method
│   ├── water_detector_sam.py   # Grounded SAM method
│   └── inpainters/
│       ├── __init__.py
│       ├── base.py             # ImageInpainter ABC
│       ├── flux_fill.py        # FluxFillPipeline wrapper
│       ├── flux_canny.py       # FluxControlPipeline (Canny) wrapper
│       ├── flux_redux.py       # FluxPriorRedux + FluxFill (visual ref)
│       └── flux_kontext.py     # FluxKontextPipeline wrapper
│
├── config/
│   ├── __init__.py
│   └── prompts.csv             # class_id, class_name, prompt rows
│
├── inputs/                     # Source images for processing
│   ├── *.jpg / *.png           # Ocean/water scene images (flat)
│   └── references/             # Reference photos per trash class
│       ├── plastic_bottle/     # class_id=0
│       ├── glass/              # class_id=1
│       ├── can/                # class_id=2
│       ├── plastic_bag/        # class_id=3
│       ├── metal_scrap/        # class_id=4
│       ├── plastic_wrapper/    # class_id=5
│       ├── trash_pile/         # class_id=6
│       └── trash/              # class_id=7
│
├── outputs/                    # fill mode outputs (generated at runtime)
│   ├── {stem}_synth.png        # Generated image
│   ├── {stem}.txt              # YOLO annotations
│   ├── {stem}_debug.png        # Bounding box overlay
│   ├── {stem}_water_mask.png   # Water mask used
│   └── generation_log.csv      # Per-image generation log
│
└── outputs_test/               # test mode outputs (generated at runtime)
    ├── canny/
    ├── redux/
    └── kontext/
```

## Key Locations

| What | Where |
|------|-------|
| CLI entry | `run.py` |
| Pipeline logic | `core/pipeline.py` |
| Add a new inpainter | `core/inpainters/` + register in `pipeline.py:load_model()` |
| Add a water detector | `core/water_detector_{name}.py` + register in `pipeline.py:_WATER_MODULES` |
| Trash class definitions | `config/prompts.csv` + `core/constants.py:OBJECT_SIZES` |
| Reference images | `inputs/references/{class_folder}/` |
| Source images | `inputs/*.jpg` |

## Naming Conventions

- **Modules:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `FluxLocalImageInpainter`, `ProcessConfig`)
- **Functions:** `snake_case` (e.g., `process_image`, `find_water_positions`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_SIDE`, `OBJECT_SIZES`)
- **Water detectors:** `core/water_detector_{method}.py` — all expose `create_water_mask()`
- **Inpainters:** `Flux{Variant}Inpainter` — all inherit from `ImageInpainter`

## Output File Naming

Given source image `{stem}.jpg`:
- `{stem}_synth.png` — generated image (fill mode)
- `{stem}_result.png` — generated image (test mode)
- `{stem}.txt` — YOLO annotations
- `{stem}_debug.png` — bounding box visualization
- `{stem}_water_mask.png` — water detection mask
