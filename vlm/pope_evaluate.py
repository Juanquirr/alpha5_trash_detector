"""
pope_evaluate.py  —  Compute POPE metrics from pope_run.py results.

Reads pope_results/pope_{model}_{tier}.csv files and produces:
  1. Console summary table (Precision / Recall / F1 / Accuracy / Yes-ratio)
  2. PNG: per-tier F1 heatmaps (classes × models) + Yes-ratio bars
  3. PNG: tier-comparison grouped bar (overall F1 and Yes-ratio across tiers)

Metrics:
    Precision  = TP / (TP + FP)    when model says YES, how often correct?
    Recall     = TP / (TP + FN)    of all positives, how many found?
    F1         = harmonic mean of P & R
    Yes-ratio  = (TP + FP) / total fraction of YES answers (hallucination bias)
    Accuracy   = (TP + TN) / total

Usage:
    python pope_evaluate.py
    python pope_evaluate.py --results pope_results/ --out pope_results/pope_eval.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pope_run import derive_pred

DEFAULT_CLASSES = [
    "container",
    "plastic",
    "metal",
    "polystyrene",
    "plastic fragment",
    "trash pile",
    "trash",
]

TIERS = ("random", "popular", "adversarial")

TIER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MODEL_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#BCB22C",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Returns dict keyed by (model, tier) → DataFrame.
    Filename pattern: pope_{model}_{tier}.csv
    """
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for f in sorted(results_dir.rglob("pope_*.csv")):
        stem = re.sub(r"^pope_", "", f.stem)   # strip leading "pope_"
        for tier in TIERS:
            if stem.endswith(f"_{tier}"):
                model = stem[: -(len(tier) + 1)]
                df    = pd.read_csv(f)
                df["label"]    = df["label"].astype(str).str.strip().str.lower()
                df["response"] = df["response"].astype(str).str.strip()
                # Derive pred from raw response (pred column no longer stored)
                df["pred"] = df.apply(
                    lambda r: derive_pred(r["response"], r["cls"]), axis=1
                )
                dfs[(model, tier)] = df
                break
    if not dfs:
        raise FileNotFoundError(f"No pope_*.csv files found in {results_dir}")
    return dfs


# ── Metrics ───────────────────────────────────────────────────────────────────

def _safe(num: int, den: int) -> float:
    return num / den * 100 if den > 0 else float("nan")


def _cell(df: pd.DataFrame) -> dict:
    """Compute TP/FP/FN/TN + all metrics for a (sub-)DataFrame."""
    pred_pos = df["pred"]  == "yes"
    true_pos = df["label"] == "yes"
    n   = len(df)
    tp  = int((pred_pos &  true_pos).sum())
    fp  = int((pred_pos & ~true_pos).sum())
    fn  = int((~pred_pos &  true_pos).sum())
    tn  = int((~pred_pos & ~true_pos).sum())
    return {
        "n":          n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":  _safe(tp, tp + fp),
        "recall":     _safe(tp, tp + fn),
        "f1":         _safe(2 * tp, 2 * tp + fp + fn),
        "accuracy":   _safe(tp + tn, n),
        "yes_ratio":  _safe(tp + fp, n),
    }


def compute_metrics(dfs: dict[tuple[str, str], pd.DataFrame],
                    classes: list[str] | None = None) -> dict:
    """
    Returns nested dict:
        metrics[model][tier] = {
            "overall":    {precision, recall, f1, accuracy, yes_ratio, n, n_timeout, ...},
            "per_class":  {cls: {precision, recall, f1, yes_ratio, n, n_timeout}}
        }
    Rows with pred="timeout" are excluded from metric computation but counted.
    """
    models = sorted({m for m, _ in dfs})
    metrics: dict = {m: {} for m in models}

    for model in models:
        for tier in TIERS:
            if (model, tier) not in dfs:
                continue
            df = dfs[(model, tier)]

            # Separate valid rows from timed-out rows (response=="TIMEOUT" is sentinel)
            n_timeout = int((df["response"] == "TIMEOUT").sum())
            valid     = df[df["response"] != "TIMEOUT"].copy()

            overall             = _cell(valid)
            overall["n_timeout"] = n_timeout
            overall["n_total"]   = len(df)   # including timeouts

            if classes is None:
                classes = DEFAULT_CLASSES
            per_class = {}
            for cls in classes:
                sub_all   = df[df["cls"] == cls]
                sub_valid = sub_all[sub_all["response"] != "TIMEOUT"]
                if not sub_all.empty:
                    cell               = _cell(sub_valid)
                    cell["n_timeout"]  = int((sub_all["response"] == "TIMEOUT").sum())
                    per_class[cls]     = cell

            metrics[model][tier] = {"overall": overall, "per_class": per_class}

    return metrics


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    models = list(metrics)
    col_w  = max(len(m) for m in models) + 2

    for tier in TIERS:
        if not any(tier in metrics[m] for m in models):
            continue
        sep = "─" * (col_w + 66)
        print(f"\n{'═'*70}")
        print(f"  POPE tier: {tier.upper()}")
        print(f"{'═'*70}")
        print(f"{'Model':<{col_w}}  {'N':>5}  {'Skip':>4}  {'Prec%':>6}  {'Rec%':>5}  "
              f"{'F1%':>5}  {'Acc%':>5}  {'Yes%':>5}")
        print(sep)
        for model in models:
            if tier not in metrics[model]:
                continue
            o = metrics[model][tier]["overall"]
            skip_note = f"{o['n_timeout']}" if o.get("n_timeout", 0) else "-"
            print(
                f"{model:<{col_w}}  {o['n']:>5}  {skip_note:>4}  "
                f"{o['precision']:>5.1f}  {o['recall']:>5.1f}  "
                f"{o['f1']:>5.1f}  {o['accuracy']:>5.1f}  {o['yes_ratio']:>5.1f}"
            )
        print(sep)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _heatmap(ax, data: np.ndarray, xlabels, ylabels, title,
             vmin=0, vmax=100, cmap="RdYlGn", fmt="{:.0f}") -> object:
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                color = "black" if 20 < v < 80 else "white"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")
            else:
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=7, color="#aaa")
    return im


def _bar_grouped(ax, models, values_per_tier, tiers_present, title, ylabel,
                 hline=None):
    x     = np.arange(len(models))
    width = 0.22
    n     = len(tiers_present)
    for ti, (tier, color) in enumerate(zip(tiers_present, TIER_COLORS)):
        offset = (ti - n / 2 + 0.5) * width
        vals   = [values_per_tier[tier].get(m, float("nan")) for m in models]
        bars   = ax.bar(x + offset, vals, width, color=color,
                        label=tier, edgecolor="white", linewidth=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold")
    if hline is not None:
        ax.axhline(hline, color="#888", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main figure ───────────────────────────────────────────────────────────────

def build_figure(metrics: dict, out: Path, classes: list[str] | None = None) -> None:
    models         = list(metrics)
    tiers_present  = [t for t in TIERS if any(t in metrics[m] for m in models)]

    if classes is None:
        classes = DEFAULT_CLASSES

    if not models or not tiers_present:
        print("No data to plot.")
        return

    n_tiers = len(tiers_present)
    # Layout: n_tiers rows of (F1-heatmap | yes-ratio-bar) + 1 bottom row (comparison)
    fig, axes = plt.subplots(
        n_tiers + 1, 2,
        figsize=(max(14, len(models) * 1.6 + 4), n_tiers * 4.2 + 4.5),
        gridspec_kw={"width_ratios": [3, 1], "hspace": 0.55, "wspace": 0.35},
    )
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "POPE Evaluation — VLM Hallucination & Per-class Detection Analysis",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Per-tier rows: F1 heatmap (left) + yes-ratio bar (right)
    for row_idx, tier in enumerate(tiers_present):
        ax_heat = axes[row_idx, 0]
        ax_bar  = axes[row_idx, 1]

        # Build (classes × models) F1 matrix
        f1_mat  = np.full((len(classes), len(models)), np.nan)
        yr_vals = []

        for j, model in enumerate(models):
            if tier not in metrics[model]:
                yr_vals.append(float("nan"))
                continue
            pc = metrics[model][tier]["per_class"]
            for i, cls in enumerate(classes):
                if cls in pc:
                    f1_mat[i, j] = pc[cls]["f1"]
            yr_vals.append(metrics[model][tier]["overall"]["yes_ratio"])

        im = _heatmap(ax_heat, f1_mat, models, classes,
                      f"[{tier}]  Per-class F1 (%)")
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.65, pad=0.01)
        cbar.set_label("F1 (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

        # Yes-ratio bar
        colors = MODEL_PALETTE[:len(models)]
        x      = np.arange(len(models))
        bars   = ax_bar.bar(x, yr_vals, color=colors,
                            edgecolor="white", linewidth=0.8, zorder=3)
        ax_bar.axhline(50, color="#888", linewidth=0.8, linestyle="--", zorder=2)
        ax_bar.text(len(models) - 0.5, 52, "50% (no bias)",
                    ha="right", va="bottom", fontsize=6.5, color="#666")
        ax_bar.set_title(f"[{tier}]  Yes-ratio (%)", fontsize=9,
                         fontweight="bold", pad=5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
        ax_bar.set_ylim(0, 115)
        ax_bar.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        ax_bar.set_axisbelow(True)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.set_ylabel("Yes answers (%)", fontsize=7.5)
        for bar, v in zip(bars, yr_vals):
            if not np.isnan(v):
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 2,
                            f"{v:.1f}%", ha="center", va="bottom",
                            fontsize=7, fontweight="bold")

    # Bottom row: tier-comparison grouped bars
    ax_f1 = axes[n_tiers, 0]
    ax_yr = axes[n_tiers, 1]

    f1_by_tier: dict[str, dict[str, float]] = {t: {} for t in tiers_present}
    yr_by_tier: dict[str, dict[str, float]] = {t: {} for t in tiers_present}
    for tier in tiers_present:
        for model in models:
            if tier in metrics[model]:
                o = metrics[model][tier]["overall"]
                f1_by_tier[tier][model] = o["f1"]
                yr_by_tier[tier][model] = o["yes_ratio"]

    _bar_grouped(ax_f1, models, f1_by_tier, tiers_present,
                 "Overall F1 by Tier\nHarder tier → more hallucination?",
                 "F1 (%)")
    _bar_grouped(ax_yr, models, yr_by_tier, tiers_present,
                 "Overall Yes-ratio by Tier\n50% = balanced; above = YES bias",
                 "Yes-ratio (%)", hline=50)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate POPE VLM results")
    parser.add_argument("--results", default="pope_results",
                        help="Directory containing pope_*.csv files")
    parser.add_argument("--questions", default="pope_questions",
                        help="Questions directory (reads metadata.json for classes)")
    parser.add_argument("--out",     default="pope_results/pope_eval.png",
                        help="Output figure path")
    args = parser.parse_args()

    # Read classes from metadata if available
    meta_path = Path(args.questions) / "metadata.json"
    if meta_path.exists():
        import json
        meta    = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta.get("classes", DEFAULT_CLASSES)
        print(f"Classes from metadata ({len(classes)}): {classes}")
    else:
        classes = DEFAULT_CLASSES

    results_dir = Path(args.results)
    print(f"Loading results from {results_dir} ...")
    dfs = load_results(results_dir)

    for (model, tier) in sorted(dfs):
        print(f"  {model}/{tier}: {len(dfs[(model, tier)])} rows")

    metrics = compute_metrics(dfs, classes)
    print_summary(metrics)
    build_figure(metrics, Path(args.out), classes)


if __name__ == "__main__":
    main()
