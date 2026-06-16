# VLM Benchmark for marine trash detection

Evaluates vision-language models on marine litter detection using POPE (binary hallucination probing), LoRA fine-tuning, and visual grounding.

## Setup

```powershell
.\vlm\setup_5.x.ps1        # .transformers-5.X-venv
.\vlm\setup_4.46.ps1       # .transformers-4.46-venv
```

| Venv | Models |
|------|--------|
| `.transformers-5.X-venv` | `smolvlm`, `smolvlm_500m`, `qwen_vl`, `qwen_2b` |
| `.transformers-4.46-venv` | `llava` |

## Models

| Key | Model | Params | VRAM | 
|-----|-------|--------|------|
| `smolvlm` | SmolVLM-Instruct | 2B | ~4 GB |
| `smolvlm_500m` | SmolVLM-500M-Instruct | 500M | ~1 GB |
| `qwen_vl` | Qwen2.5-VL-3B-Instruct | 3B | ~6 GB |
| `qwen_2b` | Qwen3-VL-2B-Instruct | 2B | ~4 GB |
| `llava` | llava-1.5-7b-hf | 7B | ~14 GB |

All models support LoRA fine-tuning.

---

## POPE evaluation

### Why POPE instead of standard detection evaluation

Standard evaluation asks "did the model correctly say YES or NO for the whole image?" That hides two very different failure modes:

1. **Misses**: trash is there, the model does not mention it (false negative)
2. **Hallucinations**: trash is NOT there, but the model invents it (false positive)

A model that hallucinates triggers constant false alarms. A model that misses fails to catch real litter. Standard accuracy cannot tell them apart.

POPE (Polling-based Object Probing Evaluation) measures both problems class by class with targeted yes/no questions.

### How it works

Instead of "describe this image", POPE asks one question per class per image:

> "Is there a rigid non-metal container such as a bottle or jar in this image? Answer yes or no."

This runs for all 7 classes on every image. We know ground truth from YOLO annotations, so each answer maps to:

| | Model says YES | Model says NO |
|---|---|---|
| **Trash present** | TP | FN (miss) |
| **Trash absent** | FP (hallucination) | TN |

### Difficulty tiers

Each tier asks all positive questions (classes present) plus an equal number of negatives (classes absent). The tier controls which negatives are selected.

**Random.** Negatives chosen at random from absent classes. Baseline with no particular pressure to hallucinate.

**Popular.** Negatives are the most frequently occurring classes in the dataset. A model biased by training frequency might "expect" containers everywhere and hallucinate them even when absent.

**Adversarial.** Negatives are classes that most often co-occur with the ground truth classes in that image. If a trash pile is present, the model is asked about classes that frequently appear alongside trash piles (e.g. plastic fragments). Semantically tempting even when absent.

Each tier uses n_neg = n_pos per image (~50/50 yes/no balance). Clean images (no annotated classes) get three negative questions to test hallucination on background-only scenes.

**Interpreting tier comparison.** If F1 and Yes-ratio stay similar across tiers, hallucinations are random. If F1 drops and Yes-ratio rises from random to adversarial, the model has semantic hallucination bias (says YES based on context, not visual evidence).

### Metrics

**Precision** = TP / (TP + FP). When the model says YES, how often is it right? Low precision means frequent false alarms.

**Recall** = TP / (TP + FN). Of all real trash, how much does the model find? Low recall means trash being missed.

**F1** = 2 TP / (2 TP + FP + FN). Harmonic mean of precision and recall. Single best summary number. A model that says YES to everything gets high recall but low precision, and F1 punishes that.

**Yes-ratio** = (TP + FP) / total. Hallucination bias detector. On a balanced dataset, a calibrated model has Yes-ratio near 50%. Much higher means systematic hallucination. Much lower means overly conservative.

---

## Scripts

### `pope_build.py`

Generates binary yes/no questions from YOLO annotations in three tiers with 50/50 balance.

```powershell
python vlm\pope_build.py --dataset ..\alpha5\datasets\alpha6
python vlm\pope_build.py --images images\
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

### `pope_run.py`

Runs yes/no questions against models. Auto-selects correct venv per model.

```powershell
python vlm\pope_run.py --model smolvlm --tier all
python vlm\pope_run.py --model all --tier all
python vlm\pope_run.py --model all --tier all --without qwen_2b,llava
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Model key, `all`, or comma-separated list |
| `--without` | none | Exclude models (with `--model all`) |
| `--tier` | `all` | `random`, `popular`, `adversarial`, or `all` |
| `--timeout` | 15 | Seconds per question (0 = disabled) |
| `--questions` | `pope_questions/` | Questions directory |
| `--images` | from metadata | Override images directory |

Output: `pope_results/pope_{model}_{tier}.csv`. Already answered questions are skipped on resume.

### `pope_evaluate.py`

Computes metrics from pope_run results and generates hallucination analysis plots.

```powershell
python vlm\pope_evaluate.py
python vlm\pope_evaluate.py --results pope_results\ --out pope_results\pope_eval.png
```

Output: F1 heatmaps, yes-ratio bars, and tier comparison chart.

### `pope_finetune_eval.py`

Three-phase pipeline: pre-eval, LoRA fine-tuning, post-eval, comparison chart.

```powershell
python vlm\pope_finetune_eval.py --model smolvlm_500m --tier all
python vlm\pope_finetune_eval.py --model all --tier all
python vlm\pope_finetune_eval.py --model smolvlm_500m --tier all --skip-pre
python vlm\pope_finetune_eval.py --model qwen_vl --tier all --max-train-samples 2000
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
| `--max-train-samples` | all | Cap training samples |
| `--timeout` | 15 | Per-question timeout (seconds) |
| `--skip-pre` | off | Skip pre-eval phase |
| `--skip-ft` | off | Skip fine-tuning phase |

Output: `pope_results/{model}_pre/` and `{model}_post/` CSVs, plus `pope_{model}_finetune_cmp.png`.

Training and eval use same images. This measures domain adaptation, not generalization.

#### How LoRA fine-tuning works

Full fine-tuning updates every weight in a model, which is prohibitively expensive for billion-parameter VLMs. Low-Rank Adaptation (LoRA) avoids this by injecting two small trainable matrices A and B alongside each frozen attention projection layer. The effective weight update is the low-rank product A × B, where the rank r (default 8) controls the capacity of the adaptation. Only these matrices are updated during training — typically less than 1% of total parameters.

For each training sample, the model receives an image and a binary question ("Is there a container in this image?"). The ground-truth answer ("yes" or "no") is appended and the loss is computed exclusively on that answer token; the image and question prefix are masked with −100 and excluded from the gradient. This forces the model to learn the visual associations for each class rather than memorising prompt patterns.

After training, the adapter matrices are merged back into the base weights (`merge_and_unload()`), producing a single model file with no inference overhead. The original adapter files (~50–200 MB) are preserved in `{out}/{model}_lora/` and can be reloaded independently of the base model.

### `grounding_eval.py`

Asks Qwen models to locate objects with bounding boxes, then compares against YOLO ground truth via IoU matching.

```powershell
python vlm\grounding_eval.py --model qwen_vl --dataset ..\alpha5\datasets\alpha6
python vlm\grounding_eval.py --model qwen_2b --dataset ..\alpha5\datasets\alpha6 --limit 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | `qwen_vl` or `qwen_2b` |
| `--dataset` | none | YOLO dataset root |
| `--images` | none | Flat image directory |
| `--labels` | none | Separate labels dir |
| `--limit` | 50 | Max images to process |
| `--iou-threshold` | 0.3 | IoU threshold for matching |

Output: `grounding_results/grounding_{model}.csv`

Only Qwen models support structured `<ref>class</ref><box>(x1,y1,x2,y2)</box>` output.

---

## Output structure

```
vlm/
├── pope_questions/
│   ├── metadata.json
│   ├── pope_random.jsonl
│   ├── pope_popular.jsonl
│   └── pope_adversarial.jsonl
├── pope_results/
│   ├── pope_{model}_{tier}.csv
│   ├── pope_eval.png
│   ├── {model}_pre/
│   ├── {model}_post/
│   └── pope_{model}_finetune_cmp.png
└── grounding_results/
    └── grounding_{model}.csv
```

## Adding a new model

1. Create `models/{name}.py` inheriting `BaseVLM` from `models/base.py`
2. Implement `load()` and `describe(image_path, prompt) -> str`
3. Register in `models/__init__.py` (`REGISTRY` and `VENV` dicts)
4. Add dependencies to `envs/requirements-*.txt`
