# INTEGRATIONS.md — External Integrations

## HuggingFace Hub

**Type:** Model registry / weights download
**Usage:** All four inpainters auto-download model weights on first instantiation
**Models:**
- `black-forest-labs/FLUX.1-Fill-dev` — Fill inpainter
- `black-forest-labs/FLUX.1-Canny-dev` — Canny inpainter
- `black-forest-labs/FLUX.1-Redux-dev` — Redux prior pipeline
- `black-forest-labs/FLUX.1-Kontext-dev` — Kontext editor

**Auth:** HuggingFace token typically required for gated models (`HUGGING_FACE_HUB_TOKEN` env var)
**Code:** `core/inpainters/flux_*.py` — `from_pretrained(...)` calls in `model_post_init`

## Roboflow

**Type:** Dataset management platform
**Usage:** Dataset upload/management (present in requirements, not in core pipeline code)
**Library:** `roboflow` package
**Auth:** Roboflow API key (env var, not yet configured in codebase)

## OpenAI API

**Type:** LLM API
**Usage:** Present in requirements (`openai` package) but no usage found in current core pipeline
**Status:** Likely planned or legacy — not wired into `run.py` or `core/`

## Autodistill / Grounded SAM

**Type:** Auto-labeling pipeline
**Usage:** `core/water_detector_sam.py` — SAM-based water mask detection method
**Library:** `autodistill-grounded-sam`
**Model:** Grounding DINO + SAM weights (downloaded automatically)

## GPU / CUDA

**Type:** Compute backend
**Usage:** All FLUX pipelines use `.to("cuda")` — requires NVIDIA GPU
**Requirements:** CUDA 12.4+, cuDNN 9
**Precision:** `torch.bfloat16` throughout (memory efficient)

## Local File System

**Input paths:**
- `inputs/` — source ocean/water images to process
- `inputs/references/{class_name}/` — reference images per trash category (Redux model)
- `config/prompts.csv` — prompt definitions per class

**Output paths:**
- `outputs/` — default fill mode output (PNG images + YOLO .txt annotations + debug overlays + water masks)
- `outputs_test/` — default test mode output (per-model subdirectories)
- `outputs/generation_log.csv` — per-image generation log

## No External Database / Auth Providers

This is a local ML pipeline tool. No database, authentication system, or webhook integrations are present.
