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

| Key | Model | Params | VRAM | Fine-tunable |
|-----|-------|--------|------|-------------|
| `smolvlm` | HuggingFaceTB/SmolVLM-Instruct | 2B | ~4 GB | ✅ LoRA SFT |
| `smolvlm_500m` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | ~1 GB | ✅ LoRA SFT |
| `qwen_vl` | Qwen/Qwen2.5-VL-3B-Instruct | 3B | ~6 GB | ✅ LoRA SFT |
| `qwen_2b` | Qwen/Qwen3-VL-2B-Instruct | 2B | ~4 GB | ✅ LoRA SFT |
| `llava_ov` | lmms-lab/llava-onevision-qwen2-0.5b-ov | 500M | ~1 GB | ✅ LoRA SFT |
| `llava` | llava-hf/llava-1.5-7b-hf | 7B | ~14 GB | ✅ LoRA SFT |
| `moondream` | vikhyatk/moondream2 | 2B | ~4 GB | ❌ trust_remote_code |
| `clip` | openai/clip-vit-large-patch14 | 307M | ~1 GB | ❌ zero-shot classifier |
| `internvl2` | OpenGVLab/InternVL2-2B | 2B | ~4 GB | ❌ trust_remote_code |

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

POPE runs three times on the same image set. Each tier asks all positive questions (classes present in the image) plus an equal number of **negative questions** (classes not present). The tier controls *which* negatives are selected — each tier picks a different subset, making the difficulty genuinely different:

#### Tier 1 — Random
Negative classes are **chosen at random** from the absent classes. This is the **baseline** — no particular pressure to hallucinate any specific class.

#### Tier 2 — Popular
Negative classes are the **most frequently occurring** in the dataset as a whole. The idea: a model biased by training frequency might "expect" plastic bottles everywhere and hallucinate them even when absent.

*Example: if plastic bottles appear in 70% of images, they are selected as negatives first. A hallucination-prone model will say YES far too often.*

#### Tier 3 — Adversarial
Negative classes are those that **most often co-occur** with the ground-truth classes in that specific image. This is the hardest tier: if a trash pile is in the image, the model is asked about classes that frequently appear alongside trash piles (e.g. plastic bags) — semantically tempting even when absent.

*Example: an image has a trash pile. The model is asked "Is there a plastic bag?" — plastic bags often appear near trash piles, so a hallucination-prone model says YES by association.*

**Balance:** each tier uses n_neg = n_pos per image (~50 % yes / 50 % no). Clean images (no annotated classes) receive three fixed negative questions to test hallucination on background-only scenes.

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

### POPE Adaptation Note

This implementation applies POPE to a **closed-set domain** of 8 trash categories rather than the original open-vocabulary COCO setting. The methodology is identical; only the class pool differs (8 vs ~80). Tier differentiation is inherently narrower with a smaller pool, so tier F1 gaps will be smaller than in the original paper. Results should be referenced as *"POPE-style evaluation adapted to a closed-set trash detection domain (8 classes)"*.

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

## POPE Fine-tuning — Teaching Models from Their Own Mistakes

`pope_finetune_eval.py` extends the POPE pipeline with a full **evaluate → fine-tune → re-evaluate → compare** loop. The goal is to measure whether a targeted adaptation on this specific trash-detection domain improves hallucination behaviour.

---

### How the Fine-tuning Works

Standard POPE evaluation reveals *where* a model fails. The fine-tuning step uses those same questions as a training signal:

1. The model sees an image and a question: *"Is there a plastic bottle in this image?"*
2. It answers YES or NO.
3. We know the ground-truth answer from YOLO annotations.
4. If the model is wrong, the error is back-propagated to adjust its weights.

**The base model is never modified.** Instead, a small set of additional weight matrices — called **LoRA adapters** — is trained on top of the frozen base. LoRA (Low-Rank Adaptation) represents the adaptation as two small matrices whose product approximates the full weight update. This means:

- Training is fast (only ~0.1–1 % of parameters are updated)
- VRAM usage is much lower than full fine-tuning
- The original model capabilities are preserved — catastrophic forgetting is avoided
- The adapters are tiny (~50–200 MB vs. several GB for the base model)

Loss is computed **only on the yes/no answer token**, not on the image tokens or the question — so the model learns to associate visual evidence with the correct binary answer rather than memorising prompt patterns.

---

### The Three-Phase Pipeline

```
[Phase 1]  Pre-eval   →  pope_results/{model}_pre/pope_{model}_{tier}.csv
[Phase 2]  LoRA SFT   →  adapters saved (merged into model for post-eval)
[Phase 3]  Post-eval  →  pope_results/{model}_post/pope_{model}_{tier}.csv
[Output]   Chart      →  pope_results/pope_{model}_finetune_cmp.png
```

The comparison chart shows per-class F1 before (blue) and after (green) fine-tuning for each tier, with delta annotations (+/−) on each bar, plus an overall metrics summary.

---

### Fine-tunable Models

| Model | Fine-tunable | Reason if not |
|-------|-------------|---------------|
| `smolvlm`, `smolvlm_500m` | ✅ | Standard HuggingFace + chat template |
| `qwen_vl`, `qwen_2b` | ✅ | Standard HuggingFace + chat template |
| `llava_ov` | ✅ | Standard HuggingFace + chat template |
| `llava` | ✅ | LlavaForConditionalGeneration, manual prompt |
| `moondream` | ❌ | Custom architecture (`trust_remote_code`) |
| `clip` | ❌ | Zero-shot classifier, not a generative model |
| `internvl2` | ❌ | Custom architecture (`trust_remote_code`) |

For non-fine-tunable models the script still runs pre-eval and generates a single-phase chart (eval-only).

---

### Running the Fine-tuning Pipeline

```powershell
# Single model — full pipeline
python vlm\pope_finetune_eval.py --model smolvlm_500m --tier all

# All models — spawns one subprocess per model (correct venv selected automatically)
python vlm\pope_finetune_eval.py --model all --tier all

# Eval-only (no fine-tuning) — useful for non-fine-tunable models or baselines
python vlm\pope_finetune_eval.py --model clip --tier all --skip-ft

# Custom training hyperparameters
python vlm\pope_finetune_eval.py --model smolvlm --epochs 5 --lora-r 16 --lr 1e-4

# Reuse existing pre-eval CSVs (skip Phase 1)
python vlm\pope_finetune_eval.py --model smolvlm_500m --skip-pre --epochs 3
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 3 | LoRA training epochs |
| `--lr LR` | 2e-4 | Learning rate |
| `--lora-r R` | 8 | LoRA rank (higher = more capacity, more VRAM) |
| `--lora-alpha A` | 16 | LoRA alpha (scaling factor) |
| `--accum N` | 4 | Gradient accumulation steps |
| `--timeout S` | 15 | Per-question inference timeout (seconds) |
| `--skip-pre` | off | Skip pre-eval if CSVs already exist |
| `--skip-ft` | off | Skip fine-tuning entirely |

> **Note on data leakage:** training and evaluation use the same image set. This is intentional — the study measures domain adaptation capability, not generalisation. Any write-up should describe this as a *"closed-loop domain adaptation experiment"*.

---

### Real-World Deployment

In a production trash-detection system the fine-tuning workflow separates cleanly into two phases:

**Development phase (GPU machine):**
```
Annotated dataset → pope_finetune_eval.py → LoRA adapters saved to {model}_lora/
```

**Deployment phase (edge device / inference server):**

Two options depending on constraints:

**Option A — Merged model (recommended for fixed deployments)**
The LoRA adapters are merged into the base model weights at export time, producing a single model file with no runtime overhead. This is the standard path for edge cameras or embedded systems where inference speed matters.

```python
# Already done automatically by pope_finetune_eval.py after training.
# To export manually:
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "smolvlm_500m_lora/")
merged = model.merge_and_unload()
merged.save_pretrained("smolvlm_500m_merged/")
```

**Option B — Hot-swappable adapters (recommended for multi-context deployments)**
Keep the base model loaded in memory and swap LoRA adapters depending on context — for example, a different adapter per camera type, distance range, or environment. Only the small adapter files need to move between deployments.

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "smolvlm_500m_lora_closeup/")
# swap adapter:
model.load_adapter("smolvlm_500m_lora_longrange/", adapter_name="longrange")
model.set_adapter("longrange")
```

**Updating in production:**
When new labelled images are collected, re-run the fine-tuning pipeline. Only the adapter files need to be re-deployed (~100 MB) — the base model (several GB) never changes.

---



```
vlm/results/
    detections_{model}.csv           Standard run results (one row per image)
    prompts.txt                      Full prompt text indexed by hash
    evaluation.png                   Standard evaluation plot

vlm/pope_questions/
    metadata.json                    Images dir + build config
    pope_random.jsonl                Binary questions, negatives sampled randomly
    pope_popular.jsonl               Binary questions, negatives = most frequent absent classes
    pope_adversarial.jsonl           Binary questions, negatives = most co-occurring absent classes

vlm/pope_results/
    pope_{model}_{tier}.csv          POPE inference results (one row per question)
    pope_eval.png                    POPE evaluation plot (all models, all tiers)

    {model}_pre/
        pope_{model}_{tier}.csv      Pre-fine-tuning POPE results
    {model}_post/
        pope_{model}_{tier}.csv      Post-fine-tuning POPE results
    pope_{model}_finetune_cmp.png    Before/after comparison chart per model
    {model}_lora/                    Saved LoRA adapter weights (if fine-tuned)
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
├── pope_build.py              Build POPE binary question JSONL files (balanced 50/50)
├── pope_run.py                POPE inference runner (venv-aware, timeout, resume)
├── pope_evaluate.py           POPE metrics + hallucination analysis plots
├── pope_finetune_eval.py      POPE eval → LoRA fine-tune → re-eval → comparison chart
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
├── pope_questions/            POPE JSONL question files + metadata
└── pope_results/              POPE CSVs, evaluation plots, LoRA adapters
```
