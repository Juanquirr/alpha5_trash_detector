"""
pope_finetune_eval.py  —  POPE evaluate → LoRA fine-tune → re-evaluate → comparison chart

Pipeline per model
──────────────────
  1. Pre-eval  : POPE inference  → pope_results/{model}_pre/pope_{model}_{tier}.csv
  2. Fine-tune : LoRA SFT (PEFT) → skipped if model not capable or peft missing
  3. Post-eval : POPE inference  → pope_results/{model}_post/pope_{model}_{tier}.csv
  4. Chart     : per-class F1 before vs after, one subplot per tier + overall summary
                 → pope_results/pope_{model}_finetune_cmp.png

Note: train and eval sets share the same images (domain adaptation study, not
generalisation benchmark). Mention data leakage in any write-up.

Fine-tunable  : smolvlm  smolvlm_500m  qwen_vl  qwen_2b  llava
Not supported : clip (zero-shot)  moondream (trust_remote_code)  internvl2

Usage
─────
    python pope_finetune_eval.py --model smolvlm_500m --tier all
    python pope_finetune_eval.py --model all --tier all
    python pope_finetune_eval.py --model clip --tier all   # eval-only
    python pope_finetune_eval.py --model smolvlm --epochs 2 --lora-r 16

Flags
─────
    --skip-pre      skip pre-eval if CSVs already exist
    --skip-ft       skip fine-tuning (chart shows pre-eval only if no post CSVs)
    --epochs N      LoRA training epochs            (default 1)
    --lr LR         learning rate                   (default 5e-5)
    --lora-r R      LoRA rank                       (default 8)
    --lora-alpha A  LoRA alpha                      (default 16)
    --accum N       gradient accumulation steps     (default 4)
    --timeout S     per-question inference timeout  (default 15)
    --questions DIR pope_questions directory        (default pope_questions)
    --images DIR    images directory                (default from metadata.json)
    --out DIR       output root                     (default pope_results)
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import REGISTRY, VENV
from pope_run import (
    load_questions,
    _already_done,
    _append_row,
    parse_yesno,
    parse_clip_score,
    derive_pred,
    _describe_with_timeout,
    _fmt_time,
    POPE_CSV_FIELDS,
    IMAGE_EXTS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

TIERS = ("random", "popular", "adversarial")

CLASSES = [
    "plastic bottle", "glass", "can", "plastic bag",
    "metal scrap", "plastic wrapper", "trash pile", "trash",
]

# Models that support LoRA SFT via PEFT
FINETUNE_CAPABLE = frozenset({
    "smolvlm", "smolvlm_500m", "qwen_vl", "qwen_2b", "llava",
})

# Chart colours
C_PRE  = "#4C72B0"   # steel blue  — before fine-tuning
C_POST = "#55A868"   # forest green — after fine-tuning
C_SKIP = "#8C8C8C"   # grey         — eval-only (no fine-tuning)


# ── Image resolution ──────────────────────────────────────────────────────────

def _resolve_image(images_dir: Path, name: str) -> Path | None:
    p = images_dir / name
    if p.exists():
        return p
    for ext in IMAGE_EXTS:
        alt = (images_dir / name).with_suffix(ext)
        if alt.exists():
            return alt
    return None


# ── Question loading ──────────────────────────────────────────────────────────

def _load_tier_questions(questions_dir: Path, tier: str) -> list[dict]:
    path = questions_dir / f"pope_{tier}.jsonl"
    if not path.exists():
        print(f"  [WARN] question file missing: {path}")
        return []
    return load_questions(path)


# ── In-process POPE evaluation ────────────────────────────────────────────────

def _run_eval_phase(
    vlm,
    model_key:    str,
    tiers:        list[str],
    questions_dir: Path,
    images_dir:   Path,
    out_dir:      Path,
    timeout_s:    float,
    phase:        str = "pre",      # "pre" or "post"
) -> None:
    """
    Run POPE evaluation for all tiers using an already-loaded VLM.
    Saves results to out_dir/pope_{model_key}_{tier}.csv.
    Resume: question_ids already in the CSV are skipped.
    """
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = vlm.device == "cuda" and torch.cuda.is_available()
    is_clip = model_key == "clip"

    for tier in tiers:
        questions = _load_tier_questions(questions_dir, tier)
        if not questions:
            continue

        csv_path  = out_dir / f"pope_{model_key}_{tier}.csv"
        done_ids  = _already_done(csv_path)
        pending   = [q for q in questions if q["question_id"] not in done_ids]

        if done_ids:
            print(f"[{model_key}/{phase}/{tier}] Resume: {len(done_ids)} done, {len(pending)} remaining.")
        if not pending:
            print(f"[{model_key}/{phase}/{tier}] Already complete.")
            continue

        print(f"[{model_key}/{phase}/{tier}] Evaluating {len(pending)} questions…")
        t_start    = time.perf_counter()
        n_timeout  = 0

        for i, q in enumerate(pending, 1):
            # Periodic cache flush to prevent allocator fragmentation on long runs.
            # Qwen3-VL tiles images dynamically; reserved-but-unallocated memory
            # grows across thousands of images and eventually causes OOM.
            if is_cuda and i % 100 == 0:
                torch.cuda.empty_cache()

            image_path = _resolve_image(images_dir, q["image"])
            if not image_path:
                continue

            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            t0                  = time.perf_counter()
            response, timed_out = _describe_with_timeout(vlm, str(image_path), q["text"], timeout_s)

            if timed_out:
                n_timeout += 1
                _append_row({
                    "question_id": q["question_id"], "image": q["image"],
                    "cls": q["cls"],   "label": q["label"],
                    "response":    "TIMEOUT",
                    "inference_s": round(time.perf_counter() - t0, 3), "vram_mb": 0,
                }, csv_path)
                continue

            if is_cuda:
                torch.cuda.synchronize()
            elapsed = round(time.perf_counter() - t0, 3)
            vram_mb = 0
            if is_cuda:
                vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

            pred = parse_clip_score(response, q["cls"]) if is_clip else parse_yesno(response)

            _append_row({
                "question_id": q["question_id"], "image": q["image"],
                "cls": q["cls"],   "label": q["label"],
                "response":    response.strip()[:300],
                "inference_s": elapsed, "vram_mb": vram_mb,
            }, csv_path)

            ok  = "✓" if pred == q["label"] else "✗"
            eta = _fmt_time((time.perf_counter() - t_start) / i * (len(pending) - i))
            print(
                f"  [{i}/{len(pending)}] {ok} {q['image'][:18]:<18s}  "
                f"{q['cls']:<16s}  gt={q['label']}  pred={pred}  "
                f"| {elapsed}s | ETA {eta}"
            )

        to_note = f"  ({n_timeout} timeouts)" if n_timeout else ""
        print(f"[{model_key}/{phase}/{tier}] Done in {_fmt_time(time.perf_counter() - t_start)}.{to_note}")

        # Release fragmented CUDA allocations between tiers so the next tier
        # doesn't OOM on a 32 GiB GPU after processing thousands of images.
        if is_cuda:
            torch.cuda.empty_cache()


# ── LoRA fine-tuning ──────────────────────────────────────────────────────────

def _build_sample_t5(vlm, image, question: str, answer: str) -> dict | None:
    """
    Build labeled training input for transformers-5.x VLMs (apply_chat_template).
    Labels mask the user+image prefix; loss is computed only on the answer token.
    Returns None on failure (sample skipped).
    """
    proc = vlm.processor
    try:
        user_msg = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
        asst_msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }

        # Prefix length: user turn + image tokens (no answer yet)
        user_text  = proc.apply_chat_template([user_msg], add_generation_prompt=True)
        user_kw    = dict(text=user_text, images=[image], return_tensors="pt")
        prefix_len = proc(**user_kw)["input_ids"].shape[1]

        # Full sequence: user + assistant answer
        full_text = proc.apply_chat_template(
            [user_msg, asst_msg], add_generation_prompt=False
        )
        inputs = proc(text=full_text, images=[image], return_tensors="pt")

        labels              = inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100   # mask prefix from loss
        inputs["labels"]    = labels
        return dict(inputs)
    except Exception:
        return None


def _build_sample_llava(vlm, image, question: str, answer: str) -> dict | None:
    """
    Build labeled training input for LLaVA 1.5 (transformers 4.46).
    Uses the USER/ASSISTANT conversation template.
    """
    proc = vlm.processor
    try:
        full_prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        inputs      = proc(text=full_prompt, images=image, return_tensors="pt")

        # Find where the ASSISTANT answer starts in the token sequence
        tok = getattr(proc, "tokenizer", None)
        if tok is None:
            return None
        answer_ids = tok.encode(f" {answer}", add_special_tokens=False)
        seq_len    = inputs["input_ids"].shape[1]
        prefix_len = seq_len - len(answer_ids)

        labels              = inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100
        inputs["labels"]    = labels
        return dict(inputs)
    except Exception:
        return None


def _build_sample(vlm, model_key: str, image, question: str, answer: str) -> dict | None:
    """Dispatch to correct sample builder based on model type."""
    if model_key == "llava":
        return _build_sample_llava(vlm, image, question, answer)
    # All other capable models use transformers-5.x chat template
    if hasattr(vlm.processor, "apply_chat_template"):
        return _build_sample_t5(vlm, image, question, answer)
    return None


def _get_lora_target_modules(model) -> list[str]:
    """
    Return the set of Linear module name-suffixes found in the model,
    excluding the language-model head and embedding layers.

    Using 'all-linear' in LoraConfig hits lm_head, which controls the full
    vocabulary distribution. Fine-tuning it at high LR collapses the output
    to a single token ('no'). This helper targets only attention + MLP
    projection layers, which are safe to adapt.
    """
    _EXCLUDE = {"lm_head", "embed_tokens", "wte", "wpe", "shared", "decoder"}
    seen: set[str] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Linear":
            suffix = name.split(".")[-1]
            if suffix not in _EXCLUDE:
                seen.add(suffix)
    targets = sorted(seen)
    if not targets:
        # Fallback: use all-linear (better than nothing)
        return "all-linear"
    return targets


def finetune_lora(
    vlm,
    model_key:   str,
    questions_dir: Path,
    images_dir:  Path,
    tiers:       list[str],
    epochs:      int   = 1,
    lr:          float = 5e-5,
    lora_r:      int   = 8,
    lora_alpha:  int   = 16,
    accum_steps: int   = 4,
) -> bool:
    """
    Fine-tune vlm.model in-place with LoRA SFT on POPE questions.
    Returns True on success, False if skipped/failed.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        print(f"[{model_key}] peft not installed — skipping fine-tuning")
        return False

    import torch
    from PIL import Image

    print(f"\n[{model_key}] === LoRA SFT: r={lora_r} α={lora_alpha} lr={lr} epochs={epochs} ===")

    # Collect unique (image, question, answer) triples across all tiers
    seen: set[tuple[str, str]] = set()
    samples_meta: list[dict]   = []
    for tier in tiers:
        for q in _load_tier_questions(questions_dir, tier):
            key = (q["image"], q["cls"])
            if key in seen:
                continue
            seen.add(key)
            img_path = _resolve_image(images_dir, q["image"])
            if img_path is None:
                continue
            samples_meta.append({
                "image_path": img_path,
                "question":   q["text"],
                "answer":     q["label"],   # ground-truth yes/no
            })

    if not samples_meta:
        print(f"[{model_key}] No training samples found — skipping fine-tuning")
        return False

    print(f"[{model_key}] Training samples: {len(samples_meta)}")

    # Apply LoRA — target attention/MLP projections only; skip lm_head to
    # prevent scrambling the output vocabulary distribution.
    target_mods = _get_lora_target_modules(vlm.model)
    print(f"[{model_key}] LoRA target modules: {target_mods}")
    lora_cfg = LoraConfig(
        r              = lora_r,
        lora_alpha     = lora_alpha,
        lora_dropout   = 0.05,
        bias           = "none",
        target_modules = target_mods,
        task_type      = TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(vlm.model, lora_cfg)
    peft_model.print_trainable_parameters()

    # Enable gradient checkpointing for large models (saves VRAM)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    if hasattr(peft_model.base_model, "gradient_checkpointing_enable"):
        try:
            peft_model.base_model.gradient_checkpointing_enable()
        except Exception:
            pass

    peft_model.train()
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(samples_meta) * epochs + accum_steps - 1) // accum_steps
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps  = max(1, total_steps // 10),
            num_training_steps = total_steps,
        )
    except Exception:
        scheduler = None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_valid    = 0
        optimizer.zero_grad()

        for step_idx, meta in enumerate(samples_meta, 1):
            try:
                image   = Image.open(meta["image_path"]).convert("RGB")
                inputs  = _build_sample(vlm, model_key, image, meta["question"], meta["answer"])
                if inputs is None:
                    continue

                device_inputs = {k: v.to(vlm.device) for k, v in inputs.items()}
                outputs = peft_model(**device_inputs)
                loss    = outputs.loss

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    continue

                (loss / accum_steps).backward()
                total_loss += loss.item()
                n_valid    += 1

                if step_idx % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] sample {step_idx} skipped — reduce batch or use smaller model")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            except Exception as e:
                continue

        # Flush remaining gradients
        if n_valid % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / n_valid if n_valid > 0 else float("nan")
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}  ({n_valid}/{len(samples_meta)} valid samples)")

    # Merge LoRA weights back into base model
    print(f"[{model_key}] Merging LoRA weights…")
    vlm.model = peft_model.merge_and_unload()
    vlm.model.eval()
    print(f"[{model_key}] Fine-tuning complete.")
    return True


# ── Metrics computation ───────────────────────────────────────────────────────

def _safe_div(a: int, b: int) -> float:
    return a / b * 100 if b > 0 else float("nan")


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom * 100 if denom > 0 else float("nan")


def compute_metrics_from_csv(csv_path: Path) -> dict:
    """
    Returns {
        "overall": {prec, rec, f1, acc, yes_ratio},
        "per_class": {cls: {f1, prec, rec}},
    }
    Returns None if file missing.
    """
    if not csv_path.exists():
        return None

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("response", "").strip() == "TIMEOUT":
                continue
            # Derive pred from raw response (pred column no longer stored)
            row["pred"] = derive_pred(row.get("response", ""), row.get("cls", ""))
            rows.append(row)

    if not rows:
        return None

    def _confusion(subset):
        tp = fp = tn = fn = 0
        for r in subset:
            lbl  = r["label"].strip().lower()
            pred = r["pred"].strip().lower()
            if   lbl == "yes" and pred == "yes": tp += 1
            elif lbl == "no"  and pred == "yes": fp += 1
            elif lbl == "no"  and pred == "no":  tn += 1
            elif lbl == "yes" and pred == "no":  fn += 1
        return tp, fp, tn, fn

    tp, fp, tn, fn = _confusion(rows)
    n = tp + fp + tn + fn
    overall = {
        "prec":      _safe_div(tp, tp + fp),
        "rec":       _safe_div(tp, tp + fn),
        "f1":        _f1(tp, fp, fn),
        "acc":       _safe_div(tp + tn, n),
        "yes_ratio": _safe_div(tp + fp, n),
    }

    per_class = {}
    for cls in CLASSES:
        sub = [r for r in rows if r["cls"].strip() == cls]
        if sub:
            tp2, fp2, tn2, fn2 = _confusion(sub)
            per_class[cls] = {
                "f1":   _f1(tp2, fp2, fn2),
                "prec": _safe_div(tp2, tp2 + fp2),
                "rec":  _safe_div(tp2, tp2 + fn2),
            }

    return {"overall": overall, "per_class": per_class}


# ── Comparison chart ──────────────────────────────────────────────────────────

def build_comparison_chart(
    model_key: str,
    pre_dir:   Path,
    post_dir:  Path | None,
    tiers:     list[str],
    out_path:  Path,
    ft_done:   bool,
) -> None:
    """
    Generate per-class F1 before/after chart for one model.
    post_dir=None or missing CSVs → eval-only chart (pre only).
    """
    n_rows = len(tiers) + 1    # one per tier + overall summary
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, n_rows * 4.2 + 1),
        gridspec_kw={"hspace": 0.55},
    )
    fig.patch.set_facecolor("#f8f9fa")

    ft_label = "Fine-tuned" if ft_done else "Eval-only (no fine-tuning)"
    fig.suptitle(
        f"POPE Fine-tuning Analysis — {model_key}  [{ft_label}]",
        fontsize=13, fontweight="bold", y=0.998,
    )

    x       = np.arange(len(CLASSES))
    width   = 0.35
    cls_labels = [c.replace(" ", "\n") for c in CLASSES]

    overall_pre_f1:  list[float] = []
    overall_post_f1: list[float] = []

    for row_idx, tier in enumerate(tiers):
        ax = axes[row_idx]

        pre_csv  = pre_dir  / f"pope_{model_key}_{tier}.csv"
        post_csv = (post_dir / f"pope_{model_key}_{tier}.csv") if post_dir else None

        pre_m  = compute_metrics_from_csv(pre_csv)
        post_m = compute_metrics_from_csv(post_csv) if (post_csv and post_csv.exists()) else None

        if pre_m is None:
            ax.set_title(f"[{tier}]  No data", fontsize=9)
            ax.axis("off")
            continue

        pre_f1  = [pre_m["per_class"].get(c, {}).get("f1",  float("nan")) for c in CLASSES]
        post_f1 = (
            [post_m["per_class"].get(c, {}).get("f1", float("nan")) for c in CLASSES]
            if post_m else None
        )

        overall_pre_f1.append(pre_m["overall"]["f1"])
        if post_m:
            overall_post_f1.append(post_m["overall"]["f1"])

        # Draw bars
        color_post = C_POST if ft_done else C_SKIP
        if post_f1 is not None:
            ax.bar(x - width / 2, pre_f1,  width, color=C_PRE,    label="Before",  alpha=0.85, edgecolor="white")
            ax.bar(x + width / 2, post_f1, width, color=color_post, label="After", alpha=0.85, edgecolor="white")
        else:
            ax.bar(x, pre_f1, width * 1.4, color=C_PRE, label="Before", alpha=0.85, edgecolor="white")

        # Delta annotations
        if post_f1 is not None:
            for xi, (pf, qf) in enumerate(zip(pre_f1, post_f1)):
                if not (np.isnan(pf) or np.isnan(qf)):
                    delta = qf - pf
                    color = "#2ca02c" if delta >= 0 else "#d62728"
                    ax.text(
                        xi + width / 2, qf + 1.5,
                        f"{delta:+.1f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color=color, fontweight="bold",
                    )

        ax.set_title(f"[{tier}]  Per-class F1 (%)", fontsize=9, fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(cls_labels, fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("F1 (%)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Overall summary row ───────────────────────────────────────────────────
    ax_sum = axes[len(tiers)]
    metrics_keys  = ["prec", "rec", "f1", "acc", "yes_ratio"]
    metrics_names = ["Precision", "Recall", "F1", "Accuracy", "Yes-ratio"]

    # Aggregate across tiers
    def _mean_metric(key: str, phase_dir: Path | None) -> list[float]:
        vals = []
        if phase_dir is None:
            return [float("nan")] * len(metrics_keys)
        for tier in tiers:
            p = phase_dir / f"pope_{model_key}_{tier}.csv"
            m = compute_metrics_from_csv(p)
            if m:
                vals.append(m["overall"].get(key, float("nan")))
        if not vals:
            return [float("nan")] * len(metrics_keys)
        return vals

    pre_overall  = [np.nanmean(_mean_metric(k, pre_dir))  for k in metrics_keys]
    post_overall = [np.nanmean(_mean_metric(k, post_dir)) for k in metrics_keys] if post_dir else None

    xm    = np.arange(len(metrics_keys))
    color_post = C_POST if ft_done else C_SKIP
    if post_overall is not None:
        ax_sum.bar(xm - width / 2, pre_overall,  width, color=C_PRE,    label="Before",  alpha=0.85, edgecolor="white")
        ax_sum.bar(xm + width / 2, post_overall, width, color=color_post, label="After", alpha=0.85, edgecolor="white")
        for xi, (pv, qv) in enumerate(zip(pre_overall, post_overall)):
            if not (np.isnan(pv) or np.isnan(qv)):
                delta = qv - pv
                color = "#2ca02c" if delta >= 0 else "#d62728"
                ax_sum.text(
                    xi + width / 2, qv + 1.5,
                    f"{delta:+.1f}",
                    ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold",
                )
    else:
        ax_sum.bar(xm, pre_overall, width * 1.4, color=C_PRE, label="Before", alpha=0.85, edgecolor="white")

    ax_sum.set_title("Overall Metrics (avg across tiers)", fontsize=9, fontweight="bold", pad=5)
    ax_sum.set_xticks(xm)
    ax_sum.set_xticklabels(metrics_names, fontsize=9)
    ax_sum.set_ylim(0, 115)
    ax_sum.set_ylabel("(%)", fontsize=8)
    ax_sum.axhline(50, color="#888", linewidth=0.7, linestyle="--")
    ax_sum.text(len(metrics_keys) - 0.45, 52, "50%", fontsize=6.5, color="#666")
    ax_sum.legend(fontsize=8, loc="upper right")
    ax_sum.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax_sum.set_axisbelow(True)
    ax_sum.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{model_key}] Chart saved → {out_path}")


# ── Single-model pipeline ─────────────────────────────────────────────────────

def run_single_model(model_key: str, args) -> None:
    import torch

    out_root   = Path(args.out)
    pre_dir    = out_root / f"{model_key}_pre"
    post_dir   = out_root / f"{model_key}_post"
    chart_path = out_root / f"pope_{model_key}_finetune_cmp.png"

    questions_dir = Path(args.questions)

    # Resolve images dir from args or metadata
    if args.images:
        images_dir = Path(args.images)
    else:
        meta_path = questions_dir / "metadata.json"
        if meta_path.exists():
            meta       = json.loads(meta_path.read_text(encoding="utf-8"))
            images_dir = Path(meta["images_dir"])
            print(f"[{model_key}] Images dir from metadata: {images_dir}")
        else:
            images_dir = Path("images")

    tiers = list(TIERS) if args.tier == "all" else [args.tier]

    # ── 1. Load model ─────────────────────────────────────────────────────────
    vlm_cls = REGISTRY[model_key]
    vlm     = vlm_cls()
    print(f"\n[{model_key}] Loading model…")
    vlm.load()

    # ── 2. Pre-eval ───────────────────────────────────────────────────────────
    if args.skip_pre and all(
        (pre_dir / f"pope_{model_key}_{t}.csv").exists() for t in tiers
    ):
        print(f"[{model_key}] Pre-eval CSVs found — skipping (--skip-pre).")
    else:
        print(f"\n[{model_key}] === Phase 1/3: PRE-EVAL ===")
        _run_eval_phase(vlm, model_key, tiers, questions_dir,
                        images_dir, pre_dir, args.timeout, phase="pre")

    # ── 3. Fine-tune ──────────────────────────────────────────────────────────
    ft_done = False
    if args.skip_ft:
        print(f"[{model_key}] Fine-tuning skipped (--skip-ft).")
    elif model_key not in FINETUNE_CAPABLE:
        print(f"[{model_key}] Not fine-tunable (zero-shot or trust_remote_code). Skipping.")
    else:
        print(f"\n[{model_key}] === Phase 2/3: FINE-TUNING ===")
        ft_done = finetune_lora(
            vlm, model_key,
            questions_dir = questions_dir,
            images_dir    = images_dir,
            tiers         = tiers,
            epochs        = args.epochs,
            lr            = args.lr,
            lora_r        = args.lora_r,
            lora_alpha    = args.lora_alpha,
            accum_steps   = args.accum,
        )

    # ── 4. Post-eval (only if fine-tuning ran) ────────────────────────────────
    if ft_done:
        print(f"\n[{model_key}] === Phase 3/3: POST-EVAL ===")
        _run_eval_phase(vlm, model_key, tiers, questions_dir,
                        images_dir, post_dir, args.timeout, phase="post")

    vlm.unload()

    # ── 5. Comparison chart ───────────────────────────────────────────────────
    has_post = ft_done and any(
        (post_dir / f"pope_{model_key}_{t}.csv").exists() for t in tiers
    )
    build_comparison_chart(
        model_key  = model_key,
        pre_dir    = pre_dir,
        post_dir   = post_dir if has_post else None,
        tiers      = tiers,
        out_path   = chart_path,
        ft_done    = ft_done,
    )


# ── Multi-model orchestrator ──────────────────────────────────────────────────

def _resolve_python(venv_name: str) -> Path:
    base = Path(__file__).parent / venv_name
    for candidate in [
        base / "Scripts" / "python.exe",
        base / "bin" / "python",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Venv '{venv_name}' not found at {base}.")


def run_all_models(model_keys: list[str], args) -> None:
    script = Path(__file__).resolve()
    errors: list[tuple[str, str]] = []
    total  = len(model_keys)

    print(f"\n{'═'*64}")
    print(f"  {total} model(s): {', '.join(model_keys)}")
    print(f"  Tier: {args.tier}  |  Epochs: {args.epochs}  |  Timeout: {args.timeout}s")
    print(f"{'═'*64}\n")

    for idx, key in enumerate(model_keys, 1):
        venv = VENV[key]
        print(f"\n[{idx}/{total}] ── {key}  (venv: {venv})")
        try:
            python = _resolve_python(venv)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            errors.append((key, "venv not found"))
            continue

        cmd = [str(python), str(script), "--model", key,
               "--tier", args.tier,
               "--epochs", str(args.epochs),
               "--lr", str(args.lr),
               "--lora-r", str(args.lora_r),
               "--lora-alpha", str(args.lora_alpha),
               "--accum", str(args.accum),
               "--timeout", str(args.timeout),
               "--questions", args.questions,
               "--out", args.out]
        if args.images:
            cmd += ["--images", args.images]
        if args.skip_pre:
            cmd.append("--skip-pre")
        if args.skip_ft:
            cmd.append("--skip-ft")

        t0     = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = _fmt_time(time.perf_counter() - t0)

        if result.returncode != 0:
            print(f"  [ERROR] {key} exited {result.returncode} after {elapsed}")
            errors.append((key, f"exit code {result.returncode}"))
        else:
            print(f"  [OK] {key} finished in {elapsed}")

    print(f"\n{'═'*64}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for k, r in errors:
            print(f"    ✗ {k}: {r}")
    else:
        print(f"  All {total} models completed.")
    print(f"{'═'*64}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="POPE evaluate → LoRA fine-tune → re-evaluate → comparison chart"
    )
    parser.add_argument("--model",      required=True,
                        help=f"Model key, 'all', or comma-separated. Options: {list(REGISTRY)}")
    parser.add_argument("--without",    default="",
                        help="Comma-separated models to exclude (with --model all)")
    parser.add_argument("--tier",       default="all",
                        choices=list(TIERS) + ["all"])
    parser.add_argument("--questions",  default="pope_questions")
    parser.add_argument("--images",     default=None)
    parser.add_argument("--out",        default="pope_results")
    parser.add_argument("--timeout",    type=float, default=15.0)
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--lora-r",     type=int,   default=8,  dest="lora_r")
    parser.add_argument("--lora-alpha", type=int,   default=16, dest="lora_alpha")
    parser.add_argument("--accum",      type=int,   default=4)
    parser.add_argument("--skip-pre",   action="store_true", dest="skip_pre")
    parser.add_argument("--skip-ft",    action="store_true", dest="skip_ft")
    args = parser.parse_args()

    exclude = {k.strip() for k in args.without.split(",") if k.strip()}

    if args.model == "all":
        keys = [k for k in REGISTRY if k not in exclude]
        run_all_models(keys, args)
        return

    keys    = [k.strip() for k in args.model.split(",")]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}.  Available: {list(REGISTRY)}")
        sys.exit(1)

    keys = [k for k in keys if k not in exclude]
    if len(keys) > 1:
        run_all_models(keys, args)
        return

    run_single_model(keys[0], args)


if __name__ == "__main__":
    main()
