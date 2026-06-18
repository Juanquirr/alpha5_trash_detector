# Marine Trash Dataset Generator

A synthetic dataset generation pipeline for floating marine debris detection. It inserts photorealistic trash objects into real harbour and ocean images using FLUX diffusion models, producing YOLO-format annotated training data.

---

## Overview

Acquiring large annotated datasets of floating marine trash is difficult — clean examples are rare, labelling is expensive, and object variety is limited. This tool automates that process by taking real harbour photographs and synthetically inserting trash at plausible water positions, producing paired images and bounding box labels ready for YOLO training.

**Output per image:**
- `*_synth.png` — synthesised image with inserted trash
- `*.txt` — YOLO bounding box annotations
- `*_debug.png` — visualisation with drawn bounding boxes
- `*_water_mask.png` — binary water detection mask
- `generation_log.csv` — per-object metadata log

---

## Requirements

- Docker with NVIDIA GPU support (CUDA 12.4)
- NVIDIA GPU with ≥16 GB VRAM (tested on RTX 5000 Ada 32 GB)
- HuggingFace account with access to the FLUX gated models

**Base image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

---

## Setup

```bash
# Build the container
docker build -t trash_generator .

# Run interactively with GPU and project mounted
docker run --gpus all -it \
  -v $(pwd):/app \
  trash_generator
```

Model weights are downloaded automatically from HuggingFace on first use. The full set of FLUX models requires approximately **60 GB** of disk space.

---

## Input Structure

```
inputs/
├── *.jpg / *.jpeg          ← harbour or ocean source images
└── references/
    ├── container/          ← reference photos per class (JPG or PNG)
    ├── plastic/
    ├── metal/
    ├── polystyrene/
    ├── trash_pile/
    └── trash/
```

---

## Usage

### Generate a full dataset

Processes all images in `inputs/` using FLUX Fill with text prompts.

```bash
python run.py fill
python run.py fill --output outputs/ --num-instances 3 --water-method hsv
```

### Compare inpainting models

Runs one or more models on a random subset of images for quality evaluation.

```bash
python run.py test --model redux --references inputs/references_composed
python run.py test --model all --max-images 5 --num-instances 1
```

**Available models:** `fill`, `canny`, `redux`, `kontext`

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num-instances` | random 2–3 | Objects inserted per image |
| `--water-method` | `hsv` | Water detection method |
| `--no-crop` | off | Inpaint full image instead of local crop |
| `--references` | `inputs/references` | Reference image directory (Redux only) |
| `--max-images` | 5 | Image limit for `test` subcommand |

---

## Inpainting Models

| Model | Strategy | Best for |
|-------|----------|----------|
| **fill** | Text-conditioned FLUX Fill | Baseline; fast; no references needed |
| **canny** | Edge-guided FluxControlPipeline | Preserving background structure |
| **redux** | Visual reference via FluxPriorRedux + Fill | Controlling object appearance |
| **kontext** | In-context editing via FluxKontext | Natural scene integration |

---

## Water Detection Methods

| Method | Strategy | Notes |
|--------|----------|-------|
| `hsv` | HSV colour range + blue dominance | Default; fast; CPU-only |
| `otsu` | Otsu threshold on brightness | Good contrast scenes |
| `kmeans` | LAB colour clustering | Robust; slower |
| `flood` | Seed-based flood fill | Lower image seeding |
| `sam` | SAM3 text-prompt segmentation | Most accurate; GPU required |

---

## Object Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `container` | Rigid non-metal container — bottles (plastic/glass), jars, rigid cups |
| 1 | `plastic` | Flat, flexible, translucent plastic — bags, film, soft wrappers |
| 2 | `metal` | Specular metallic reflection — cans, aluminium foil, metal scrap |
| 3 | `polystyrene` | White opaque matte foam — EPS blocks, polystyrene cups/plates |
| 4 | `trash_pile` | Dense cluster of multiple objects — indistinguishable mixed waste |
| 5 | `trash` | Single unclassifiable item — pallets, pellets, fauna, glass, other |

---

## Redux Reference Preparation

Redux produces better results when reference images show the object already in a marine context. Two helper scripts are provided for this:

### 1. Segment objects from reference photos

Uses SAM3 to cut objects out of background-cluttered photos, saving transparent RGBA PNGs.

```bash
python scripts/segment_references.py
python scripts/segment_references.py --class can --det-threshold 0.2
```

Output: `inputs/references/{class}/*.png` (RGBA, transparent background)

### 2. Compose objects onto water backgrounds

Pastes segmented objects onto real water patches extracted from the input images, producing RGB reference images that show each object floating in a contextually realistic scene.

```bash
python scripts/prepare_redux_references.py
python scripts/prepare_redux_references.py --class can --n 8
```

Output: `inputs/references_composed/{class}/*.jpg`

Then pass the composed directory to Redux:

```bash
python run.py test --model redux --references inputs/references_composed
```

---

## Pipeline

```
inputs/*.jpeg
    │
    ├─ prepare_image()          Resize to ≤1024px, round to multiples of 16
    ├─ create_water_mask()      Detect water regions (binary mask)
    ├─ find_water_positions()   Sample N object positions inside water
    │
    └─ for each position:
        ├─ compute_crop_region()    Extract 320–640px context crop
        ├─ create_mask()            Soft elliptical inpaint mask
        ├─ model.inpaint()          Generate object in masked region
        ├─ paste crop back          Composite result into full image
        └─ compute_yolo_bbox()      Normalised bounding box annotation
```

---

## Project Structure

```
generator/
├── run.py                        CLI entry point
├── requirements.txt
├── config/
│   └── prompts.csv               Text prompts per class (4–5 per class)
├── core/
│   ├── pipeline.py               Main orchestrator
│   ├── constants.py              Sizes, margins, crop parameters
│   ├── image_utils.py            Resize, mask, crop, bbox, debug overlay
│   ├── water_detector*.py        Pluggable water detection methods
│   └── inpainters/
│       ├── flux_fill.py          FLUX Fill inpainter
│       ├── flux_canny.py         FLUX Canny inpainter
│       ├── flux_redux.py         FLUX Redux inpainter
│       └── flux_kontext.py       FLUX Kontext inpainter
├── scripts/
│   ├── segment_references.py     SAM3 object segmentation helper
│   ├── prepare_redux_references.py  Water-background compositing helper
│   └── water_masks.py            Debug: generate water masks for all inputs
└── inputs/
    ├── *.jpeg                    Source harbour/ocean images
    └── references/               Per-class reference images
```
