# STACK.md — Technology Stack

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

`run.py` — argparse CLI with two subcommands:
- `fill` — full dataset generation with FLUX Fill
- `test` — model comparison mode (canny/redux/kontext/all)

## Docker Details

- **Base:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **System deps:** `libgl1-mesa-glx`, `libglib2.0-0`, `git`
- **Port:** 8001 (FastAPI, not currently activated in CMD)
- **CMD:** `/bin/bash` (interactive shell — intended for manual invocation)
- **PYTHONUNBUFFERED:** `1`
