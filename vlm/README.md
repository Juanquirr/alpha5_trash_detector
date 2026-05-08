# Vision-Language Model Benchmark

Evaluates 9 VLMs on trash detection: accuracy, inference time, VRAM usage, and hallucination analysis via POPE. Outputs per-image CSVs and comparative evaluation plots.

---

## Requirements

- Windows (setup scripts are PowerShell)
- Python 3.10+
- NVIDIA GPU with at least 2 GB VRAM (model sizes range from 1 GB to 14 GB)
- Two separate virtual environments (transformers 5.x and 4.46.x are incompatible)

---

## Environment Setup

Run once on the execution machine:

```powershell
.\vlm\setup_5.x.ps1          # creates .transformers-5.X-venv
.\vlm\setup_4.46.ps1   # creates .transformers-4.46-venv
```

| Venv | transformers | Models |
|------|-------------|--------|
| `.transformers-5.X-venv` | ≥5.0 | `smolvlm`, `smolvlm_500m`, `qwen_vl`, `qwen_2b`, `llava_ov` |
| `.transformers-4.46-venv` | 4.46.3 | `llava`, `moondream`, `clip`, `internvl2` |

---

## Model Lineup

| Key | Model | Params | VRAM |
|-----|-------|--------|------|
| `smolvlm` | HuggingFaceTB/SmolVLM-Instruct | 2B | ~4 GB |
| `smolvlm_500m` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | ~1 GB |
| `qwen_vl` | Qwen/Qwen2.5-VL-3B-Instruct | 3B | ~6 GB |
| `qwen_2b` | Qwen/Qwen3-VL-2B-Instruct | 2B | ~4 GB |
| `llava_ov` | lmms-lab/llava-onevision-qwen2-0.5b-ov | 500M | ~1 GB |
| `llava` | llava-hf/llava-1.5-7b-hf | 7B | ~14 GB |
| `moondream` | vikhyatk/moondream2 | 2B | ~4 GB |
| `clip` | openai/clip-vit-large-patch14 | 307M | ~1 GB |
| `internvl2` | OpenGVLab/InternVL2-2B | 2B | ~4 GB |

---

## Running Models

`run.py` auto-selects the correct venv per model — no manual activation needed.

```powershell
# All models
python vlm\run.py --model all --images images\

# Single model
python vlm\run.py --model smolvlm --images images\

# Subset
python vlm\run.py --model smolvlm,internvl2 --images images\

# Limit images for smoke-testing
python vlm\run.py --model all --images images\ --limit 20

# Structured JSON output mode (per-class counts instead of DETECTED/CLEAN)
python vlm\run.py --model smolvlm --images images\ --mode json
```

**Resume:** if `results/detections_{model}.csv` already exists, processed images are skipped. Delete the CSV to force a full re-run.

---

## Detection Logic

Two prompt modes available via `--mode`:

| Mode | Prompt style | Parser |
|------|-------------|--------|
| `text` (default) | Describe scene → `DETECTED: class, class` or `CLEAN` | Regex, longest-match-first |
| `json` | Fill fixed 8-key JSON with per-class counts | JSON parser |

The text parser uses longest-match-first with consume to avoid substring bugs (e.g. "trash" matching inside "trash pile").

---

## Standard Evaluation

```powershell
pip install pandas matplotlib
python vlm\evaluate.py --images images\ --results vlm\results\ --out eval.png
```

Reads YOLO `.txt` annotations from `images/`. Generates a 3-row plot:

- **Row 1:** Accuracy / Precision / Recall per model (binary: garbage vs. clean)
- **Row 2:** F1 / Inference Time / VRAM per model
- **Row 3:** Per-class recall heatmap (8 classes × all models)

> Include clean images in your test set. If every image contains trash, Precision = 100% trivially and binary metrics are meaningless.

---

## POPE Evaluation — Understanding if Models Actually See What They Say

### The Problem with Standard Evaluation

Standard evaluation asks: *"did the model correctly say YES or NO for the whole image?"*

That is useful, but it hides two very different types of failure:

1. **Misses** — trash is there, the model doesn't mention it *(false negative)*
2. **Hallucinations** — trash is NOT there, but the model invents it *(false positive)*

These failures have completely different causes and completely different consequences for a real deployment. A model that hallucinates a lot will trigger constant false alarms. A model that misses a lot will fail to catch real litter. Standard accuracy cannot tell them apart.

POPE (Polling-based Object Probing Evaluation) is a methodology designed specifically to measure both problems, class by class, with a focused yes/no question format.

---

### How POPE Works — The Core Idea

Instead of asking *"describe this image"*, POPE asks one targeted question per class per image:

> *"Is there a plastic bottle in this image? Answer with yes or no."*

This is done for **all 8 classes** on **every image**. The model answers YES or NO. We already know the ground truth from the YOLO annotations. So for each question we know whether the answer was:

| | Model says YES | Model says NO |
|---|---|---|
| **Trash actually present** | ✅ True Positive (TP) | ❌ False Negative (FN) — **miss** |
| **Trash NOT present** | ❌ False Positive (FP) — **hallucination** | ✅ True Negative (TN) |

By accumulating these counts across all images and all questions, we get precise metrics per model and per class.

---

### The Three Difficulty Tiers

POPE runs three times, with the same questions but in different ordering of the *negative* questions (classes NOT present in the image). This tests whether the model is more likely to hallucinate certain types of objects:

#### Tier 1 — Random
Negative questions appear in random order. This is the **baseline** — no particular pressure to hallucinate any specific class.

#### Tier 2 — Popular
Negative questions are ordered by **class frequency in the dataset** — most common classes first. The idea: a model that has seen lots of images with plastic bottles might "expect" plastic bottles everywhere and hallucinate them even when they're absent. This tier specifically tests that bias.

*Example: if plastic bottles appear in 70% of images, they are asked first as negatives. A hallucination-prone model will say YES far too often.*

#### Tier 3 — Adversarial
Negative questions are ordered by **co-occurrence with the classes actually present** in each image. This is the hardest tier: if a trash pile is in the image, the model is asked first about classes that frequently appear alongside trash piles (e.g. plastic bags). These are semantically "tempting" because the context makes them plausible.

*Example: an image has a trash pile. The model is asked "Is there a plastic bag?" — plastic bags often appear near trash piles, so a poor model might say YES by association even if none is visible.*

**If a model's F1 and Yes-ratio stay similar across all three tiers** → hallucinations are random/unsystematic.
**If F1 drops and Yes-ratio rises from random → adversarial** → the model has a semantic hallucination bias (it says YES based on context, not actual visual evidence).

---

### Metrics Explained

#### Precision — *When the model says YES, is it right?*
```
Precision = TP / (TP + FP)
```
- High precision = few false alarms
- Low precision = the model cries wolf — it says "there's a can here" when there isn't

#### Recall — *Does the model find all the real trash?*
```
Recall = TP / (TP + FN)
```
- High recall = the model catches most real instances
- Low recall = trash is being missed — dangerous for a detection system

#### F1 — *The balanced summary*
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
The harmonic mean of Precision and Recall. If you need a single number, use F1. A model that says YES to everything gets Recall=100% but Precision=very low → F1 punishes this.

#### Yes-ratio — *How often does the model say YES overall?*
```
Yes-ratio = (TP + FP) / total questions
```
This is the **hallucination bias detector**. A perfectly calibrated model on a balanced dataset would have Yes-ratio ≈ 50% (since half the questions are positive, half negative). If a model has Yes-ratio = 80%, it is saying YES far too often — it is hallucinating things that aren't there.

- Yes-ratio ≈ 50% → balanced, no systematic bias
- Yes-ratio >> 50% → model has a strong YES bias (hallucination-prone)
- Yes-ratio << 50% → model is overly conservative (misses real trash)

---

### Running POPE

**Step 1 — Build questions** (only once per image set):
```powershell
python vlm\pope_build.py --images images\
```
Reads YOLO `.txt` annotations. Writes three JSONL files to `vlm\pope_questions\`:
- `pope_random.jsonl`
- `pope_popular.jsonl`
- `pope_adversarial.jsonl`

Each line is one binary question:
```json
{"question_id": 1, "image": "foto1.jpg", "cls": "plastic bottle",
 "text": "Is there a plastic bottle in this image? Answer with yes or no.",
 "label": "yes"}
```

Also writes `pope_questions\metadata.json` so step 2 finds images automatically.

**Step 2 — Run inference:**
```powershell
# One model, all three tiers
python vlm\pope_run.py --model smolvlm --tier all

# All models, all tiers (auto-selects correct venv per model)
python vlm\pope_run.py --model all --tier all

# Exclude slow models
python vlm\pope_run.py --model all --tier all --without qwen_2b,llava

# Custom timeout (skip question if inference exceeds N seconds)
python vlm\pope_run.py --model all --tier all --timeout 30

# Disable timeout
python vlm\pope_run.py --model smolvlm --tier random --timeout 0
```

Results are written to `vlm\pope_results\pope_{model}_{tier}.csv`. Resume is automatic — already-answered question IDs are skipped.

**Step 3 — Evaluate:**
```powershell
python vlm\pope_evaluate.py
```
Prints a per-tier summary table and saves `vlm\pope_results\pope_eval.png`:

- **Per-tier F1 heatmaps** — classes × models, green = good, red = poor
- **Per-tier Yes-ratio bars** — shows hallucination bias, dashed line at 50%
- **Tier comparison** — F1 and Yes-ratio grouped by tier, reveals if adversarial tier hurts specific models

---

### How to Read the Results

**A good model** for trash detection deployment shows:
- F1 > 70% on all three tiers
- Yes-ratio near 50% (not systematically hallucinating)
- F1 stays stable from random → adversarial tier (no semantic bias)

**Red flags:**
- Yes-ratio >> 60%: model is guessing YES too often — will trigger false alarms constantly
- F1 drops significantly from random → adversarial: model confuses semantically related classes
- Recall very low on specific classes (e.g. metal scrap = 10%): model systematically misses that class regardless of prompt

**CLIP note:** CLIP does not generate text — it ranks candidates by similarity score. POPE handles CLIP differently: instead of parsing yes/no from generated text, it checks whether the similarity score for the queried class exceeds 0.25. Results are valid but not directly comparable to generative models.

---

## Output Files

```
vlm/results/
    detections_{model}.csv     Standard run results (one row per image)
    prompts.txt                Full prompt text indexed by hash
    evaluation.png             Standard evaluation plot

vlm/pope_questions/
    metadata.json              Images dir + build config
    pope_random.jsonl          Binary questions, negatives in random order
    pope_popular.jsonl         Binary questions, negatives by class frequency
    pope_adversarial.jsonl     Binary questions, negatives by co-occurrence

vlm/pope_results/
    pope_{model}_{tier}.csv    POPE run results (one row per question)
    pope_eval.png              POPE evaluation plot
```

---

## Adding a New Model

1. Create `models/{name}.py` inheriting `BaseVLM` from `models/base.py`
2. Implement `load()` and `describe(image_path, prompt) -> str` (lazy imports inside `load()`)
3. Register in `models/__init__.py`: add to `REGISTRY` and `VENV` dicts
4. Add dependencies to the correct `envs/requirements-*.txt`

---

## Directory Structure

```
vlm/
├── run.py                     CLI entry point + venv orchestrator
├── results.py                 CSV writer with prompt_hash dedup
├── evaluate.py                Standard evaluation plots from GT + CSVs
├── pope_build.py              Build POPE binary question JSONL files
├── pope_run.py                POPE inference runner (venv-aware, timeout, resume)
├── pope_evaluate.py           POPE metrics + hallucination analysis plots
├── setup_5.x.ps1              Creates .transformers-5.X-venv
├── setup_4.46.ps1             Creates .transformers-4.46-venv
├── envs/
│   ├── requirements-5x.txt    transformers 5.x deps
│   └── requirements-compat.txt transformers 4.46.3 deps
├── models/
│   ├── base.py                BaseVLM, DETECTION_PROMPT, parse_response()
│   ├── __init__.py            REGISTRY + VENV dicts
│   ├── smolvlm.py             HuggingFaceTB/SmolVLM-Instruct (2B)
│   ├── smolvlm_500m.py        HuggingFaceTB/SmolVLM-500M-Instruct (500M)
│   ├── qwen_vl.py             Qwen/Qwen2.5-VL-3B-Instruct (3B)
│   ├── qwen2b.py              Qwen/Qwen3-VL-2B-Instruct (2B)
│   ├── llava_ov.py            lmms-lab/llava-onevision-qwen2-0.5b-ov (500M)
│   ├── llava.py               llava-hf/llava-1.5-7b-hf (7B)
│   ├── moondream.py           vikhyatk/moondream2 (2B)
│   ├── clip.py                openai/clip-vit-large-patch14 (zero-shot, 307M)
│   └── internvl2.py           OpenGVLab/InternVL2-2B (2B)
├── results/                   Standard run CSVs
├── pope_questions/            POPE JSONL question files
└── pope_results/              POPE inference CSVs + evaluation plot
```
