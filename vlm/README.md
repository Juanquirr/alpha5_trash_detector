# vlm — Vision-Language Model Benchmark

Evaluates 10 VLMs on trash detection accuracy, inference time, and VRAM usage. Outputs per-image CSVs and a comparative evaluation plot.

For full implementation notes, known issues, and architecture decisions see [CONTEXT.md](CONTEXT.md).

---

## Requirements

- Windows (setup scripts are PowerShell)
- Python 3.10+
- NVIDIA GPU (models range from 1 GB to 24 GB VRAM)
- Two separate virtual environments (transformers 5.x and 4.46.x are incompatible)

---

## Environment Setup

Run once on the execution machine:

```powershell
.\vlm\setup.ps1          # creates .transformers-5.X-venv
.\vlm\setup_compat.ps1   # creates .transformers-4.46-venv
```

Both scripts install PyTorch with CUDA via `--index-url` before the `requirements-*.txt` deps.

| Venv | transformers | Models |
|------|-------------|--------|
| `.transformers-5.X-venv` | ≥5.0 | `smolvlm`, `qwen_vl`, `videollama3` |
| `.transformers-4.46-venv` | 4.46.3 | `moondream`, `llava`, `blip2`, `instructblip`, `clip`, `paligemma`, `idefics`, `mplug_owl3` |

> **PaliGemma** requires a HuggingFace login (gated model): `huggingface-cli login`

---

## Running Models

`run.py` auto-selects the correct venv via subprocess — no manual activation needed when running `--model all`.

```powershell
# All models (auto-switches venv per model)
python vlm\run.py --model all --images images\

# Single model (activate the correct venv first)
python vlm\run.py --model smolvlm --images images\

# Comma-separated subset
python vlm\run.py --model smolvlm,qwen_vl --images images\

# Limit images for smoke-testing
python vlm\run.py --model all --images images\ --limit 50
```

**Resume:** if `results/detections_{model}.csv` already exists, processed images are skipped automatically. Delete the CSV to force a full re-run.

---

## Detection Logic

Prompt strategy: describe-first (chain-of-thought). The model describes the scene before classifying. Reduces false negatives vs. direct YES/NO prompting.

Expected response format:
```
[Scene description...]
DETECTED: plastic bottle, can
```
or `CLEAN` if no waste found.

The parser in `models/base.py:parse_response` applies in order:
1. Find last `DETECTED:` line → extract class names after it
2. Find `CLEAN` → `garbage_detected = False`
3. Fallback: scan full response text for class names

---

## Output Format

Each model writes `results/detections_{model}.csv`:

| Column | Content |
|--------|---------|
| `timestamp` | ISO datetime |
| `image` | filename |
| `model` | model key |
| `variant` | HuggingFace model ID |
| `prompt_hash` | SHA1[:8] of prompt text |
| `response` | raw model output (re-parseable without re-running) |
| `garbage_detected` | `True` / `False` |
| `classes_detected` | comma-separated class names |
| `inference_s` | seconds per image |
| `vram_mb` | peak VRAM MB |

Full prompt text is stored in `results/prompts.txt` indexed by hash.

---

## Evaluation

```powershell
pip install pandas matplotlib
python vlm\evaluate.py --images images\ --results vlm\results\ --out eval.png
```

Reads YOLO `.txt` annotations from `images/` (same stem as image file). Generates a 3-row plot:

- Row 1: Accuracy / Precision / Recall per model
- Row 2: F1 / Inference Time / VRAM per model
- Row 3: Per-class recall heatmap (models × 8 classes)

> Include clean images (no waste) in your test set. If all images contain trash, Precision = 100% for all models trivially and binary metrics are meaningless.

---

## Adding a New Model

1. Create `models/{name}.py` inheriting `BaseVLM` from `models/base.py`
2. Implement `load()` and `describe(image_path, prompt) -> str` with lazy imports inside `load()`
3. Register in `models/__init__.py`: add to `REGISTRY` dict and `VENV` dict
4. Add dependencies to the correct `envs/requirements-*.txt`

---

## Directory Structure

```
vlm/
├── run.py                         CLI entry point + venv orchestrator
├── results.py                     CSV writer with prompt_hash dedup
├── evaluate.py                    Evaluation plots from GT + CSVs
├── setup.ps1                      Creates .transformers-5.X-venv
├── setup_compat.ps1               Creates .transformers-4.46-venv
├── VLMs.tsv                       Model metadata reference table
├── envs/
│   ├── requirements-5x.txt        transformers 5.x deps
│   └── requirements-compat.txt    transformers 4.46.3 deps
├── models/
│   ├── base.py                    BaseVLM, DETECTION_PROMPT, parse_response()
│   ├── __init__.py                REGISTRY + VENV dicts
│   ├── smolvlm.py                 HuggingFaceTB/SmolVLM-Instruct
│   ├── qwen_vl.py                 Qwen/Qwen2.5-VL-3B-Instruct
│   ├── moondream.py               vikhyatk/moondream2
│   ├── llava.py                   llava-hf/llava-1.5-7b-hf
│   ├── blip2.py                   Salesforce/blip2-flan-t5-xl
│   ├── instructblip.py            Salesforce/instructblip-flan-t5-xl
│   ├── clip.py                    openai/clip-vit-large-patch14 (zero-shot)
│   ├── paligemma.py               google/paligemma-3b-mix-448 (gated)
│   ├── idefics.py                 HuggingFaceM4/idefics2-8b
│   ├── mplug_owl3.py              mPLUG/mPLUG-Owl3-7B-240728
│   └── videollama3.py             BLOCKED — VideoInput missing from releases
└── results/                       Output CSVs (gitignored except *.csv)
```
