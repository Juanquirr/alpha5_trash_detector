# Alpha5 — Marine Waste Detection Platform

<div align="center">

[![ULPGC](https://img.shields.io/badge/ULPGC-TFG%202025--2026-004B97)](https://www.ulpgc.es/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLO26](https://img.shields.io/badge/Model-YOLO26-purple)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://www.docker.com/)
[![LICENSE](https://img.shields.io/badge/License-Proprietary-red)](LICENSE.md)

**Alpha5** is an end-to-end platform for marine and urban waste detection,
covering synthetic data generation, automatic labelling, multi-strategy YOLO
inference, and Vision-Language Model evaluation.

</div>

---

## Bachelor's Thesis (TFG)

This repository contains the source code of the following **Bachelor's Final Project (TFG)**:

> **AI-Based System for Waste Detection in Maritime and Coastal Environments**
> *(Sistema basado en inteligencia artificial para la detección de residuos en entornos marítimos-costeros)*

| | |
|---|---|
| **University** | [Universidad de Las Palmas de Gran Canaria (ULPGC)](https://www.ulpgc.es/) |
| **Degree** | Bachelor's Degree in Computer Engineering (*Grado en Ingeniería Informática*) |
| **School** | Escuela de Ingeniería Informática (EII) |
| **Academic year** | 2025–2026 |

### Author

| Name | GitHub |
|------|--------|
| Juan Carlos Rodríguez Ramírez | [@Juanquirr](https://github.com/Juanquirr) |

### Supervisors

| Name |
|------|
| Nelson Monzón López |
| Pablo Fernández Moniz |

---

## Repository Structure

This monorepo unifies four previously independent components into a single
pipeline. Each module can run independently and has its own Dockerfile.

```
alpha5_trash_detector/
├── alpha5/          Detection model — training, inference, GUI visualizer
├── generator/       Synthetic dataset generation with FLUX inpainting
├── vlm/             Vision-Language Model evaluation benchmark
├── LICENSE.md
└── README.md
```

---

## Pipeline Overview

All modules share the same **6 waste categories** (shape-first taxonomy):

| ID | Class | Description | Includes | Excludes |
|----|-------|-------------|----------|---------|
| 0 | `container` | Rigid non-metal container with recognisable shape | PET/glass bottles, jars, pots, tubes (toothpaste, tomato), rigid/transparent plastic cups, plastic food cans, drums, jerrycans | Foam EPS cups → `polystyrene`; crushed shapeless containers → `trash`; metal cans/tins → `metal` |
| 1 | `plastic` | Flexible, flat, amorphous plastic | Plastic bags, transparent film, wrappers, shrink wrap, bubble wrap, carrier bags | Rigid plastic → `container`; small fragments → `trash` |
| 2 | `metal` | Object with specular metallic reflection | Soda/beer cans, metal food tins, aluminium foil, metal scrap, metal caps, pull-rings | Plastic cans → `container`; unidentifiable metal objects → `trash` |
| 3 | `polystyrene` | EPS foam material — white opaque matte, spongy texture | McDonald's/coffee foam cups, foam plates, supermarket foam trays, expanded polystyrene blocks, white cork | Rigid white plastic cups → `container`; crushed shapeless foam → `trash` |
| 4 | `trash_pile` | Dense cluster of multiple mixed objects | Groups where ≥3 objects are visible together with no possibility of separating individual bboxes | Any single object of any class → its own class |
| 5 | `trash` | Single item unclassifiable into any above class | Cardboard/paper (paper cups, bread bags, boxes), pallets, pellets, wood, fabric, marine fauna, deformed shapeless objects, glass, cigarettes | Any identifiable object with clear shape/material |

### Class tiebreaker rules

- **`container` vs `polystyrene`** → check texture. Matte spongy foam = `polystyrene`. Smooth/shiny/transparent = `container`.
- **Crushed/deformed object** → if material is still identifiable, keep its class. If shape and material are both unrecognisable → `trash`.
- **Any remaining doubt** → `trash` by default.

```
                          ┌──────────────┐
  Real harbour images ──► │  generator/  │ ──► Synthetic training images
                          │  FLUX models │     + YOLO annotations
                          └──────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
  Labelled dataset ─────► │   alpha5/    │ ──► Trained model (.pt)
                          │  YOLO train  │     + inference results
                          │  + 6 methods │
                          └──────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
  Test images ──────────► │    vlm/      │ ──► Accuracy/speed/VRAM
                          │  5 VLMs      │     comparison report
                          └──────────────┘
```

---

## Modules

### Alpha5 — Detection (`alpha5/`)

YOLO26-based detection system with six configurable inference strategies and an
interactive GUI for visual comparison.

**Components:**

| Directory | Purpose |
|-----------|---------|
| `train/` | Model training + hyperparameter tuning |
| `datasets/scripts/` | Dataset preparation (stratified split, YOLO↔COCO conversion) |
| `tests/experiments/` | Inference methods + benchmarking |
| `tests/visualizer/` | Interactive GUI for method comparison |

#### Training

```bash
# Dataset preparation — instance-stratified split
python alpha5/datasets/scripts/img_stratifier.py /path/to/data \
  --output /path/to/output --train 0.7 --val 0.2 --test 0.1

# Training
python alpha5/train/train_yolo.py /path/to/data.yaml yolo26x.pt \
  --epochs 300 --batch -1 --imgsz 640 --patience 15 --device 0

# Hyperparameter tuning (Bayesian optimization)
python alpha5/train/hyperparam_yolo_tunning.py /path/to/data.yaml yolo26x.pt 50 20
```

#### Inference Methods

| Method | Speed | Small Objects | Large Objects | Use Case |
|--------|-------|---------------|---------------|----------|
| Basic | 1.0x | Fair | Excellent | Real-time, baseline |
| Tiled | 1.5-2.5x | Good | Good | High-res, dense scenes |
| MultiScale | 3.0-4.0x | Excellent | Good | Variable object sizes |
| TTA | 4.0-6.0x | Fair | Good | Orientation/lighting variability |
| SuperRes | 1.1-1.3x | Good | Good | Low-quality imagery |
| **Hybrid** | **2.5-3.5x** | **Excellent** | **Excellent** | **Research, validation** |

#### GUI

```bash
python alpha5/tests/visualizer/run_visualizer.py
```

Side-by-side method comparison with zoom, pan, collapsible panels, and
adjustable confidence/IoU thresholds.

#### Docker

```bash
docker build -f alpha5/Dockerfile -t alpha5:latest .
docker run -it --gpus all --shm-size=8g \
  -v "$(pwd):/ultralytics/USER" alpha5:latest
```

---

### Generator — Synthetic Data (`generator/`)

Synthetic marine trash dataset generator using FLUX diffusion models. Inserts
photorealistic trash objects into real harbour/ocean photographs, producing
paired images and YOLO annotations.

**Inpainting models:** Fill (text-conditioned), Canny (edge-guided),
Redux (visual reference), Kontext (in-context editing).

**Water detection:** HSV, Otsu, KMeans, Flood fill, SAM3.

```bash
# Generate dataset
python generator/run.py fill --output outputs/ --num-instances 3

# Compare models
python generator/run.py test --model all --max-images 5
```

Requires NVIDIA GPU with 16+ GB VRAM. Full documentation in
[generator/README.md](generator/README.md).

#### Docker

```bash
docker build -f generator/Dockerfile -t trash_generator:latest .
docker run --gpus all -it -v $(pwd)/generator:/app trash_generator
```

---

### VLM Evaluation (`vlm/`)

Benchmark framework evaluating Vision-Language Models on trash detection using
POPE (binary hallucination probing) and LoRA fine-tuning.

**Models:** SmolVLM-Instruct (2B), SmolVLM-500M-Instruct, Qwen2.5-VL-3B-Instruct,
Qwen3-VL-2B-Instruct, LLaVA-1.5-7B.

```bash
# Build POPE questions from dataset
python vlm/pope_build.py --dataset alpha5/datasets/alpha10

# Run evaluation
python vlm/pope_run.py --model all --tier all

# LoRA fine-tuning + comparison
python vlm/pope_finetune_eval.py --model qwen3_vl --tier all
```

Requires two virtual environments (transformers 5.x and 4.46.x). Setup scripts
and full documentation in [vlm/README.md](vlm/README.md).

---

## Requirements

Each module has independent dependencies. Common baseline:

- Python 3.8+
- PyTorch with CUDA support
- Docker with NVIDIA GPU support (recommended)

| Module | Key Dependencies | GPU VRAM |
|--------|-----------------|----------|
| alpha5 | Ultralytics, OpenCV, Tkinter | 8+ GB |
| generator | diffusers, FLUX models | ~32 GB |
| vlm | transformers (5.x + 4.46) | 1-16 GB |

---

## Citation

```bibtex
@thesis{alpha5_tfg_2026,
  author      = {Rodr\'{i}guez Ram\'{i}rez, Juan Carlos},
  title       = {Sistema basado en inteligencia artificial para la detecci\'{o}n
                 de residuos en entornos mar\'{i}timos-costeros},
  school      = {Universidad de Las Palmas de Gran Canaria},
  year        = {2026},
  type        = {Trabajo de Fin de Grado},
  department  = {Escuela de Ingenier\'{i}a Inform\'{a}tica},
  note        = {Grado en Ingenier\'{i}a Inform\'{a}tica.
                 \url{https://github.com/Juanquirr/alpha5_trash_detector}}
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

This project uses a **Proprietary License** for non-commercial use only.
Commercial use, production deployment, or redistribution requires prior written
permission from the author. See [LICENSE](LICENSE.md) for details.
