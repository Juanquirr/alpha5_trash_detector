# VLM Garbage Detector — Project Context

## Goal
Evaluate multiple Vision-Language Models (VLMs) for garbage/litter detection and classification in images. Compare accuracy, inference time, and VRAM usage across models.

## Hardware (execution machine)
- GPU: RTX Ada 5000, 32 GB VRAM
- OS: Windows Server
- Workflow: code edited on staging machine → `git push` → execution machine `git pull` and runs

---

## Project Structure

```
vlm_detector/
├── models/
│   ├── base.py            # BaseVLM abstract class, DETECTION_PROMPT, parse_response()
│   ├── __init__.py        # REGISTRY + VENV dicts
│   ├── smolvlm.py         # HuggingFaceTB/SmolVLM-Instruct
│   ├── moondream.py       # vikhyatk/moondream2
│   ├── llava.py           # llava-hf/llava-1.5-7b-hf
│   ├── blip2.py           # Salesforce/blip2-flan-t5-xl
│   ├── instructblip.py    # Salesforce/instructblip-flan-t5-xl
│   ├── clip.py            # openai/clip-vit-large-patch14 (zero-shot, no text gen)
│   ├── paligemma.py       # google/paligemma-3b-mix-448 (needs HF token)
│   ├── idefics.py         # HuggingFaceM4/idefics2-8b
│   ├── mplug_owl3.py      # mPLUG/mPLUG-Owl3-7B-240728
│   ├── qwen_vl.py         # Qwen/Qwen2.5-VL-3B-Instruct
│   └── videollama3.py     # BLOCKED — VideoInput missing from transformers releases
├── envs/
│   ├── requirements-5x.txt       # transformers 5.x deps
│   └── requirements-compat.txt   # transformers 4.46.x deps
├── results/               # output CSVs (gitignored except *.csv)
├── run.py                 # CLI runner + venv orchestrator
├── results.py             # CSV writer (prompt_hash dedup)
├── evaluate.py            # evaluation graphs from YOLO GT + CSVs
├── setup.ps1              # creates .transformers-5.X-venv
└── setup_compat.ps1       # creates .transformers-4.46-venv
```

---

## Virtual Environments

| Venv | transformers | Models |
|------|-------------|--------|
| `.transformers-5.X-venv` | 5.x | smolvlm, qwen_vl, videollama3 |
| `.transformers-4.46-venv` | 4.46.3 | moondream, llava, blip2, instructblip, clip, paligemma, idefics, mplug_owl3 |

Setup (run once on execution machine):
```powershell
.\setup.ps1          # 5.X venv
.\setup_compat.ps1   # 4.46 venv
```

---

## Running Models

```powershell
# Single model (activate correct venv first):
python run.py --model smolvlm --images images\

# All models — auto-switches venv via subprocess, no manual activation needed:
python run.py --model all --images images\

# Comma-separated subset:
python run.py --model smolvlm,qwen_vl --images images\

# Limit for testing:
python run.py --model all --images images\ --limit 100
```

**Resume:** if `results/detections_{model}.csv` exists, already-processed images are skipped automatically.

---

## Detection Logic

**Prompt strategy:** describe-first (chain-of-thought) → model describes the scene → then classifies. Reduces false negatives vs direct YES/NO.

**8 classes (YOLO IDs):**

| ID | Class | Key discriminator |
|----|-------|------------------|
| 0 | plastic bottle | plastic container WITH visible cap/lid |
| 1 | glass | glass BOTTLE with neck shape (not jars) |
| 2 | can | cylindrical metal can, any condition |
| 3 | plastic bag | bag shape, large dimensions |
| 4 | metal scrap | small metal/aluminium litter (tuna cans, sprays, foil) — NOT structural metal |
| 5 | plastic wrapper | small snack/candy wrapping, flatter than bag |
| 6 | trash pile | accumulated heap of mixed waste |
| 7 | trash | catch-all for unclassifiable waste |

**Expected response format:**
```
[Scene description...]
DETECTED: plastic bottle, can
```
or `CLEAN` if no garbage.

**Parser** (`base.py:parse_response`):
1. Finds last `DETECTED:` → extracts classes after it
2. Finds `CLEAN` → False
3. Fallback: scans full text for class names

---

## Output CSV (`results/detections_{model}.csv`)

| Column | Content |
|--------|---------|
| timestamp | ISO datetime |
| image | filename |
| model | model key |
| variant | HuggingFace model ID |
| prompt_hash | SHA1[:8] of prompt (full text in `results/prompts.txt`) |
| response | raw model output — re-parseable without re-running |
| garbage_detected | True/False |
| classes_detected | comma-separated detected classes |
| inference_s | seconds per image |
| vram_mb | peak VRAM MB |

---

## Evaluation

```powershell
pip install pandas matplotlib
python evaluate.py --images images\ --results results\ --out results\eval.png
```

Reads YOLO `.txt` annotations from `images/` (same name as image, YOLO format).
Generates `evaluation.png` with:
- Row 1: Accuracy / Precision / Recall
- Row 2: F1 / Inference Time / VRAM
- Row 3: Per-class recall heatmap (models × classes)

---

## Known Issues / Decisions

- **VideoLLaMA3**: blocked — custom code requires `VideoInput` from `transformers.image_utils` not present in any release. Raises `RuntimeError` on `load()`.
- **CLIP**: zero-shot only, no text generation. Overrides `detect_garbage()` fully. Response stores probability scores per class.
- **PaliGemma**: requires `huggingface-cli login` (gated model).
- **BLIP2**: was broken (float16 cast on input_ids + OPT prompt leakage). Fixed by switching to `blip2-flan-t5-xl` and per-key `.to(device)`.
- **Moondream**: uses `torch_dtype` (not `dtype`) — transformers 4.46 compat.
- **Dataset bias**: if test set has no clean images, Precision = 100% for all models trivially. Include clean images for meaningful binary metrics.

---

## Adding a New Model

1. Create `models/{name}.py` inheriting `BaseVLM`, lazy imports inside `load()`
2. Implement `load()` and `describe(image_path, prompt) -> str`
3. Add to `REGISTRY` and `VENV` in `models/__init__.py`
4. Add dependencies to `envs/requirements-5x.txt` or `envs/requirements-compat.txt`
