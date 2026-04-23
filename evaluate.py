"""
VLM Evaluation Script
=====================
Compares results across all detections_*.csv files in results/.

Usage:
    python evaluate.py                          # performance metrics only
    python evaluate.py --labels labels.csv      # + accuracy/precision/recall
    python evaluate.py --gen-labels             # generate labels template then exit
    python evaluate.py --out results/eval.png   # custom output path
    python evaluate.py --results-dir results/   # custom results directory
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for f in sorted(results_dir.glob("detections_*.csv")):
        model = re.sub(r"^detections_", "", f.stem)
        df = pd.read_csv(f)
        df["garbage_detected"] = df["garbage_detected"].astype(str).str.upper() == "TRUE"
        dfs[model] = df
    if not dfs:
        raise FileNotFoundError(f"No detections_*.csv files found in {results_dir}")
    return dfs


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df["label"] = df["label"].astype(str).str.upper() == "TRUE"
    return df[["image", "label"]]


def generate_labels_template(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    images = sorted(set(img for df in dfs.values() for img in df["image"]))
    pd.DataFrame({"image": images, "label": ""}).to_csv(out, index=False)
    print(f"Labels template written to {out}")
    print("Fill the 'label' column with True (garbage) or False (clean), then re-run with --labels.")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(dfs: dict[str, pd.DataFrame], labels_df: pd.DataFrame | None) -> dict:
    metrics = {}
    for model, df in dfs.items():
        m: dict = {}
        m["n"] = len(df)
        m["time_mean"] = df["inference_s"].mean()
        m["time_std"] = df["inference_s"].std()
        m["vram_mean"] = df["vram_mb"].mean()
        m["detection_rate"] = df["garbage_detected"].mean() * 100

        if labels_df is not None:
            merged = df.merge(labels_df, on="image", how="inner")
            n = len(merged)
            if n == 0:
                m["accuracy"] = m["precision"] = m["recall"] = m["f1"] = float("nan")
            else:
                pred = merged["garbage_detected"].astype(bool)
                true = merged["label"].astype(bool)
                tp = int((pred & true).sum())
                fp = int((pred & ~true).sum())
                fn = int((~pred & true).sum())
                tn = int((~pred & ~true).sum())
                m["accuracy"]  = (tp + tn) / n * 100
                m["precision"] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
                m["recall"]    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
                m["f1"]        = (
                    2 * tp / (2 * tp + fp + fn) * 100
                    if (2 * tp + fp + fn) > 0 else 0.0
                )
                m["n_matched"] = n

        metrics[model] = m
    return metrics


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _bar(ax, models, values, colors, title, ylim=None, ylabel="", fmt=".1f"):
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%{fmt}"))
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim or max(values) * 0.02) * 0.02,
                f"{v:{fmt}}",
                ha="center", va="bottom", fontsize=7.5,
            )


def _bar_err(ax, models, means, stds, colors, title, ylabel=""):
    bars = ax.bar(
        models, means, yerr=stds, color=colors,
        edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#333"},
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main plot ─────────────────────────────────────────────────────────────────

def build_figure(metrics: dict, has_gt: bool, out: Path) -> None:
    models = list(metrics.keys())
    colors = PALETTE[: len(models)]

    if has_gt:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("VLM Garbage Detector — Model Comparison", fontsize=15, fontweight="bold", y=1.01)

        # Row 0: accuracy metrics
        _bar(axes[0][0], models,
             [metrics[m]["accuracy"] for m in models],
             colors, "Accuracy (%)", ylim=105, ylabel="%")
        _bar(axes[0][1], models,
             [metrics[m]["precision"] for m in models],
             colors, "Precision (%)", ylim=105, ylabel="%")
        _bar(axes[0][2], models,
             [metrics[m]["recall"] for m in models],
             colors, "Recall (%)", ylim=105, ylabel="%")

        # Row 1: performance metrics
        _bar_err(axes[1][0], models,
                 [metrics[m]["time_mean"] for m in models],
                 [metrics[m]["time_std"] for m in models],
                 colors, "Inference Time — mean ± std", ylabel="seconds")
        _bar(axes[1][1], models,
             [metrics[m]["vram_mean"] for m in models],
             colors, "VRAM Usage — mean", ylabel="MB", fmt=".0f")
        _bar(axes[1][2], models,
             [metrics[m]["detection_rate"] for m in models],
             colors, "Detection Rate (% images flagged)", ylim=105, ylabel="%")

    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "VLM Garbage Detector — Performance Comparison\n"
            "(no ground truth — run with --labels for accuracy metrics)",
            fontsize=13, fontweight="bold",
        )
        _bar_err(axes[0], models,
                 [metrics[m]["time_mean"] for m in models],
                 [metrics[m]["time_std"] for m in models],
                 colors, "Inference Time — mean ± std", ylabel="seconds")
        _bar(axes[1], models,
             [metrics[m]["vram_mean"] for m in models],
             colors, "VRAM Usage — mean", ylabel="MB", fmt=".0f")
        _bar(axes[2], models,
             [metrics[m]["detection_rate"] for m in models],
             colors, "Detection Rate (% images flagged)", ylim=105, ylabel="%")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(metrics: dict, has_gt: bool) -> None:
    col_w = max(len(m) for m in metrics) + 2
    header_base = f"{'Model':<{col_w}}  {'Images':>7}  {'Time(s)':>9}  {'VRAM(MB)':>9}  {'Det%':>6}"
    header_gt   = "  Acc%   Prec%  Rec%   F1%"
    print("\n" + "─" * (len(header_base) + (len(header_gt) if has_gt else 0)))
    print(header_base + (header_gt if has_gt else ""))
    print("─" * (len(header_base) + (len(header_gt) if has_gt else 0)))

    for model, m in metrics.items():
        line = (
            f"{model:<{col_w}}  {m['n']:>7}  "
            f"{m['time_mean']:>7.2f}s  {m['vram_mean']:>8.0f}  {m['detection_rate']:>5.1f}%"
        )
        if has_gt:
            line += (
                f"  {m['accuracy']:>4.1f}  {m['precision']:>5.1f}  "
                f"{m['recall']:>4.1f}  {m['f1']:>4.1f}"
            )
        print(line)
    print("─" * (len(header_base) + (len(header_gt) if has_gt else 0)))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM garbage detector results")
    parser.add_argument("--results-dir", default="results", help="Directory with detections_*.csv")
    parser.add_argument("--labels", default=None, help="CSV with columns: image, label (True/False)")
    parser.add_argument("--gen-labels", action="store_true", help="Generate labels template and exit")
    parser.add_argument("--out", default="results/evaluation.png", help="Output figure path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dfs = load_results(results_dir)
    print(f"Loaded {len(dfs)} model(s): {', '.join(dfs)}")

    if args.gen_labels:
        generate_labels_template(dfs, Path(args.labels or "results/labels.csv"))
        return

    labels_df = None
    has_gt = False
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.exists():
            labels_df = load_labels(labels_path)
            has_gt = True
            print(f"Ground truth loaded: {len(labels_df)} images from {labels_path}")
        else:
            print(f"Warning: --labels file not found ({labels_path}). Running without GT.")

    metrics = compute_metrics(dfs, labels_df)
    print_summary(metrics, has_gt)
    build_figure(metrics, has_gt, Path(args.out))


if __name__ == "__main__":
    main()
