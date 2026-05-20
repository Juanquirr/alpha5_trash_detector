# Alpha5 — Marine Waste Detection Platform

<div align="center">

[![ULPGC](https://img.shields.io/badge/ULPGC-TFG%202025--2026-004B97)](https://www.ulpgc.es/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv26](https://img.shields.io/badge/Model-YOLOv26-purple)](https://github.com/ultralytics/ultralytics)
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

**Commit history** for each component is preserved in dedicated branches:

| Branch | Component | Commits |
|--------|-----------|---------|
| `main` | Unified monorepo + original Alpha5 history | full |
| `history/vlm-detector` | VLM evaluation framework | 10+ |
| `history/trash-generator` | FLUX-based data generator | 30+ |

---

## Pipeline Overview

All modules share the same **7 waste categories** (shape-first taxonomy):

| ID | Class | Description |
|----|-------|-------------|
| 0 | `container` | Rigid non-metal container — bottles (plastic/glass), jars, rigid cups |
| 1 | `plastic` | Flat, flexible, translucent plastic — bags, film, soft wrappers |
| 2 | `metal` | Specular metallic reflection — cans, aluminium foil, metal scrap |
| 3 | `polystyrene` | White opaque matte foam — EPS blocks, polystyrene cups/plates |
| 4 | `plastic_fragment` | Small, compact, rigid 3D plastic — caps, fragments, cutlery, straws |
| 5 | `trash_pile` | Dense cluster of multiple objects — indistinguishable mixed waste |
| 6 | `trash` | Single unclassifiable item — pallets, pellets, fauna, glass, other |

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
                          │  10 VLMs     │     comparison report
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

Benchmark framework evaluating 10 Vision-Language Models on trash detection
accuracy, inference time, and VRAM usage.

**Models evaluated:** SmolVLM, Qwen2.5-VL, Moondream, LLaVA, BLIP2,
InstructBLIP, CLIP, PaliGemma, mPLUG-Owl3, VideoLLaMA3.

```bash
# Run all models
python vlm/run.py --model all --images images/

# Single model
python vlm/run.py --model smolvlm --images images/

# Evaluation report
python vlm/evaluate.py --images images/ --results vlm/results/ --out eval.png
```

Requires two virtual environments (transformers 5.x and 4.46.x). Setup scripts
and full documentation in [vlm/CONTEXT.md](vlm/CONTEXT.md).

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
