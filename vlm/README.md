# VLM Benchmark for Trash Detection

Evaluates 8 vision-language models on marine litter detection using binary POPE questions, LoRA fine-tuning, and visual grounding.

---

## Setup

```powershell
.\vlm\setup_5.x.ps1        # .transformers-5.X-venv
.\vlm\setup_4.46.ps1       # .transformers-4.46-venv
```

| Venv | Models |
|------|--------|
| `.transformers-5.X-venv` | `smolvlm`, `smolvlm_500m`, `qwen_vl`, `qwen_2b` |
| `.transformers-4.46-venv` | `llava`, `moondream`, `clip`, `internvl2` |

---

## Models

| Key | Model | Params | VRAM | LoRA |
|-----|-------|--------|------|------|
| `smolvlm` | SmolVLM-Instruct | 2B | ~4 GB | Yes |
| `smolvlm_500m` | SmolVLM-500M-Instruct | 500M | ~1 GB | Yes |
| `qwen_vl` | Qwen2.5-VL-3B-Instruct | 3B | ~6 GB | Yes |
| `qwen_2b` | Qwen3-VL-2B-Instruct | 2B | ~4 GB | Yes |
| `llava` | llava-1.5-7b-hf | 7B | ~14 GB | Yes |
| `moondream` | moondream2 | 2B | ~4 GB | No |
| `clip` | clip-vit-large-patch14 | 307M | ~1 GB | No |
| `internvl2` | InternVL2-2B | 2B | ~4 GB | No |

---

## Scripts

### 1. `run.py` — Standard detection

Asks each model to describe images and parses detected classes.

```powershell
python vlm\run.py --model all --images images\
python vlm\run.py --model smolvlm --images images\ --limit 20
python vlm\run.py --model smolvlm --images images\ --mode json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Model key, `all`, or comma-separated list |
| `--images` | required | Directory with images |
| `--limit N` | all | Process only first N images |
| `--mode` | `text` | `text` (free description) or `json` (structured counts) |

Output: `vlm/results/detections_{model}.csv`
Resume: existing rows skipped automatically.

---

### 2. `evaluate.py` — Standard evaluation plots

Compares model detections against YOLO ground truth.

```powershell
python vlm\evaluate.py --images images\ --results vlm\results\ --out eval.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | required | Directory with YOLO `.txt` annotations |
| `--results` | `vlm/results/` | Directory with detection CSVs |
| `--out` | `evaluation.png` | Output plot path |

Output: accuracy, precision, recall, F1, inference time, VRAM, and per-class recall heatmap.

---

### POPE Evaluation — Understanding if Models Actually See What They Say

#### The Problem with Standard Evaluation

Standard evaluation asks: *"did the model correctly say YES or NO for the whole image?"*

That is useful, but it hides two very different types of failure:

1. **Misses** — trash is there, the model doesn't mention it *(false negative)*
2. **Hallucinations** — trash is NOT there, but the model invents it *(false positive)*

These failures have completely different causes and completely different consequences for a real deployment. A model that hallucinates a lot will trigger constant false alarms. A model that misses a lot will fail to catch real litter. Standard accuracy cannot tell them apart.

POPE (Polling-based Object Probing Evaluation) is a methodology designed specifically to measure both problems, class by class, with a focused yes/no question format.

#### How POPE Works — The Core Idea

Instead of asking *"describe this image"*, POPE asks one targeted question per class per image:

> *"Is there a rigid non-metal container such as a bottle or jar in this image? Answer yes or no."*

This is done for **all 7 classes** on **every image**. The model answers YES or NO. We already know the ground truth from the YOLO annotations. So for each question we know whether the answer was:

| | Model says YES | Model says NO |
|---|---|---|
| **Trash actually present** | ✅ True Positive (TP) | ❌ False Negative (FN) — **miss** |
| **Trash NOT present** | ❌ False Positive (FP) — **hallucination** | ✅ True Negative (TN) |

By accumulating these counts across all images and all questions, we get precise metrics per model and per class.

#### The Three Difficulty Tiers

POPE runs three times on the same image set. Each tier asks all positive questions (classes present in the image) plus an equal number of **negative questions** (classes not present). The tier controls *which* negatives are selected — each tier picks a different subset, making the difficulty genuinely different:

**Tier 1 — Random**
Negative classes are **chosen at random** from the absent classes. This is the **baseline** — no particular pressure to hallucinate any specific class.

**Tier 2 — Popular**
Negative classes are the **most frequently occurring** in the dataset as a whole. The idea: a model biased by training frequency might "expect" containers everywhere and hallucinate them even when absent.

*Example: if containers appear in 70% of images, they are selected as negatives first. A hallucination-prone model will say YES far too often.*

**Tier 3 — Adversarial**
Negative classes are those that **most often co-occur** with the ground-truth classes in that specific image. This is the hardest tier: if a trash pile is in the image, the model is asked about classes that frequently appear alongside trash piles (e.g. plastic fragments) — semantically tempting even when absent.

*Example: an image has a trash pile. The model is asked "Is there a plastic fragment?" — plastic fragments often appear near trash piles, so a hallucination-prone model says YES by association.*

**Balance:** each tier uses n_neg = n_pos per image (~50 % yes / 50 % no). Clean images (no annotated classes) receive three fixed negative questions to test hallucination on background-only scenes.

**If a model's F1 and Yes-ratio stay similar across all three tiers** → hallucinations are random/unsystematic.
**If F1 drops and Yes-ratio rises from random → adversarial** → the model has a semantic hallucination bias (it says YES based on context, not actual visual evidence).

#### Metrics Explained

**Precision — When the model says YES, is it right?**
```
Precision = TP / (TP + FP)
```
- High precision = few false alarms
- Low precision = the model cries wolf — it says "there's metal here" when there isn't

**Recall — Does the model find all the real trash?**
```
Recall = TP / (TP + FN)
```
- High recall = the model catches most real instances
- Low recall = trash is being missed — dangerous for a detection system

**F1 — The balanced summary**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
The harmonic mean of Precision and Recall. If you need a single number, use F1. A model that says YES to everything gets Recall=100% but Precision=very low → F1 punishes this.

**Yes-ratio — How often does the model say YES overall?**
```
Yes-ratio = (TP + FP) / total questions
```
This is the **hallucination bias detector**. A perfectly calibrated model on a balanced dataset would have Yes-ratio ≈ 50% (since half the questions are positive, half negative). If a model has Yes-ratio = 80%, it is saying YES far too often — it is hallucinating things that aren't there.

- Yes-ratio ≈ 50% → balanced, no systematic bias
- Yes-ratio >> 50% → model has a strong YES bias (hallucination-prone)
- Yes-ratio << 50% → model is overly conservative (misses real trash)

---

### 3. `pope_build.py` — Build POPE questions

Generates binary yes/no questions from YOLO annotations. Three tiers (random, popular, adversarial) with 50/50 yes/no balance.

```powershell
# From YOLO dataset (recommended)
python vlm\pope_build.py --dataset ..\alpha5\datasets\alpha6

# From flat directory
python vlm\pope_build.py --images images\

# Separate labels directory
python vlm\pope_build.py --images images\ --labels labels\
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | none | YOLO dataset root (train/val/test splits) |
| `--images` | `images/` | Flat image directory (ignored if `--dataset`) |
| `--labels` | none | Separate labels dir (used with `--images`) |
| `--out` | `pope_questions/` | Output directory |
| `--seed` | 42 | Random seed |

Output: `pope_questions/pope_{random,popular,adversarial}.jsonl` + `metadata.json`

---

### 4. `pope_run.py` — POPE inference

Runs yes/no questions against models. Auto-selects correct venv per model.

```powershell
python vlm\pope_run.py --model smolvlm --tier all
python vlm\pope_run.py --model all --tier all
python vlm\pope_run.py --model all --tier all --without qwen_2b,llava
python vlm\pope_run.py --model smolvlm --tier random --timeout 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Model key, `all`, or comma-separated list |
| `--without` | none | Exclude models (with `--model all`) |
| `--tier` | `all` | `random`, `popular`, `adversarial`, or `all` |
| `--timeout` | 15 | Seconds per question (0 = no timeout) |
| `--questions` | `pope_questions/` | Questions directory |
| `--images` | from metadata | Override images directory |

Output: `pope_results/pope_{model}_{tier}.csv`
Resume: answered question IDs skipped automatically.

---

### 5. `pope_evaluate.py` — POPE metrics and plots

Reads all CSV results and generates hallucination analysis.

```powershell
python vlm\pope_evaluate.py
```

Output: `pope_results/pope_eval.png` with F1 heatmaps, yes-ratio bars, and tier comparison.

---

### 6. `pope_finetune_eval.py` — Evaluate, fine-tune, re-evaluate

Three-phase pipeline: pre-eval, LoRA SFT, post-eval, comparison chart.

```powershell
# Full pipeline
python vlm\pope_finetune_eval.py --model smolvlm_500m --tier all

# All models
python vlm\pope_finetune_eval.py --model all --tier all

# Skip pre-eval (reuse existing CSVs)
python vlm\pope_finetune_eval.py --model smolvlm_500m --tier all --skip-pre

# Eval only, no fine-tuning
python vlm\pope_finetune_eval.py --model clip --tier all --skip-ft

# Cap training samples for faster runs
python vlm\pope_finetune_eval.py --model qwen_vl --tier all --max-train-samples 2000

# Custom hyperparameters
python vlm\pope_finetune_eval.py --model smolvlm --epochs 3 --lora-r 16 --lr 1e-4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Model key, `all`, or comma-separated list |
| `--without` | none | Exclude models (with `--model all`) |
| `--tier` | `all` | Tier(s) to evaluate |
| `--epochs` | 1 | LoRA training epochs |
| `--lr` | 5e-5 | Learning rate |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--accum` | 4 | Gradient accumulation steps |
| `--max-train-samples` | all | Cap training to N random samples (faster) |
| `--timeout` | 15 | Per-question inference timeout (seconds) |
| `--skip-pre` | off | Skip pre-eval phase |
| `--skip-ft` | off | Skip fine-tuning phase |
| `--questions` | `pope_questions/` | Questions directory |
| `--images` | from metadata | Override images directory |

Output:
- `pope_results/{model}_pre/` and `{model}_post/` CSVs
- `pope_results/pope_{model}_finetune_cmp.png` comparison chart

Note: training and eval use same images. This measures domain adaptation, not generalization.

---

### 7. `grounding_eval.py` — Visual grounding (proof-of-concept)

Asks Qwen models to locate objects with bounding boxes, compares against YOLO GT via IoU.

```powershell
python vlm\grounding_eval.py --model qwen_vl --dataset ..\alpha5\datasets\alpha6
python vlm\grounding_eval.py --model qwen_2b --dataset ..\alpha5\datasets\alpha6 --limit 50
python vlm\grounding_eval.py --model qwen_vl --images imgs\ --labels lbls\ --limit 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | `qwen_vl` or `qwen_2b` only |
| `--dataset` | none | YOLO dataset root |
| `--images` | none | Flat image directory |
| `--labels` | none | Separate labels dir |
| `--limit` | all | Max images to process |
| `--iou-threshold` | 0.3 | IoU threshold for matching |

Output: `grounding_results/grounding_{model}.csv`

Only Qwen models support structured `<ref>class</ref><box>(x1,y1,x2,y2)</box>` output.

---

## Output structure

```
vlm/
├── results/
│   └── detections_{model}.csv           run.py output
├── pope_questions/
│   ├── metadata.json                    Image paths + build config
│   ├── pope_random.jsonl
│   ├── pope_popular.jsonl
│   └── pope_adversarial.jsonl
├── pope_results/
│   ├── pope_{model}_{tier}.csv          pope_run.py output
│   ├── pope_eval.png                    pope_evaluate.py plot
│   ├── {model}_pre/                     Pre-fine-tuning CSVs
│   ├── {model}_post/                    Post-fine-tuning CSVs
│   └── pope_{model}_finetune_cmp.png    Comparison chart
└── grounding_results/
    └── grounding_{model}.csv            grounding_eval.py output
```

---

## Adding a new model

1. Create `models/{name}.py` inheriting `BaseVLM` from `models/base.py`
2. Implement `load()` and `describe(image_path, prompt) -> str`
3. Register in `models/__init__.py` (`REGISTRY` and `VENV` dicts)
4. Add dependencies to `envs/requirements-*.txt`
