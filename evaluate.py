"""
VLM Evaluation Script
=====================
Reads YOLO annotations from images/ and compares against detections_*.csv results.

Usage:
    python evaluate.py                             # images in images/, results in results/
    python evaluate.py --images images/ --results results/
    python evaluate.py --out results/eval.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Class maps ────────────────────────────────────────────────────────────────

YOLO_ID_TO_CLASS = {
    0: "plastic bottle",
    1: "glass",
    2: "can",
    3: "plastic bag",
    4: "metal scrap",      # not in VLM classes — kept in GT, skipped in class comparison
    5: "plastic wrapper",
    6: "trash pile",
    7: "trash",
}

# Classes the VLMs can actually predict (no metal scrap)
VLM_COMPARABLE_CLASSES = [
    "plastic bottle", "glass", "can", "plastic bag",
    "metal scrap", "plastic wrapper", "trash pile", "trash",
]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ── YOLO label loading ────────────────────────────────────────────────────────

def load_yolo_labels(images_dir: Path) -> pd.DataFrame:
    """Parse all .txt YOLO annotations in images_dir.

    Returns DataFrame: image | gt_detected | gt_classes (frozenset of class names)
    Empty .txt  = clean image (gt_detected=False).
    Missing .txt = image skipped (not included).
    """
    rows = []
    for txt in sorted(images_dir.glob("*.txt")):
        image_name = txt.with_suffix(".jpg").name  # assume .jpg; also check .png below
        # resolve actual image extension
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            candidate = images_dir / txt.with_suffix(ext).name
            if candidate.exists():
                image_name = candidate.name
                break

        content = txt.read_text().strip()
        if not content:
            rows.append({"image": image_name, "gt_detected": False, "gt_classes": frozenset()})
            continue

        class_ids = set()
        for line in content.splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    pass

        gt_classes = frozenset(YOLO_ID_TO_CLASS[i] for i in class_ids if i in YOLO_ID_TO_CLASS)
        rows.append({"image": image_name, "gt_detected": bool(gt_classes), "gt_classes": gt_classes})

    if not rows:
        raise FileNotFoundError(f"No .txt annotation files found in {images_dir}")

    df = pd.DataFrame(rows)
    print(f"Ground truth: {len(df)} images  |  "
          f"{df['gt_detected'].sum()} with garbage  |  "
          f"{(~df['gt_detected']).sum()} clean")
    return df


# ── Results loading ───────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for f in sorted(results_dir.glob("detections_*.csv")):
        model = re.sub(r"^detections_", "", f.stem)
        df = pd.read_csv(f)
        df["garbage_detected"] = df["garbage_detected"].astype(str).str.upper() == "TRUE"
        df["classes_detected"] = df["classes_detected"].fillna("").astype(str)
        dfs[model] = df
    if not dfs:
        raise FileNotFoundError(f"No detections_*.csv files found in {results_dir}")
    print(f"Models loaded: {', '.join(dfs)}")
    return dfs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(dfs: dict[str, pd.DataFrame], gt: pd.DataFrame) -> dict:
    metrics = {}

    for model, df in dfs.items():
        m: dict = {}
        m["n_total"] = len(df)
        m["time_mean"] = df["inference_s"].mean()
        m["time_std"]  = df["inference_s"].std()
        m["vram_mean"] = df["vram_mb"].mean()
        m["detection_rate"] = df["garbage_detected"].mean() * 100

        merged = df.merge(gt, on="image", how="inner")
        n = len(merged)
        m["n_matched"] = n

        if n == 0:
            m["accuracy"] = m["precision"] = m["recall"] = m["f1"] = float("nan")
            m["class_recall"] = {}
            metrics[model] = m
            continue

        pred = merged["garbage_detected"].astype(bool)
        true = merged["gt_detected"].astype(bool)
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        tn = int((~pred & ~true).sum())

        m["accuracy"]  = (tp + tn) / n * 100
        m["precision"] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        m["recall"]    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        m["f1"]        = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0.0

        # Per-class recall: did VLM mention the class when GT says it's present?
        class_recall = {}
        for cls in VLM_COMPARABLE_CLASSES:
            gt_has = merged["gt_classes"].apply(lambda s: cls in s)
            if gt_has.sum() == 0:
                continue  # class not present in any image for this model's subset
            pred_has = merged["classes_detected"].str.lower().str.contains(
                re.escape(cls), na=False
            )
            class_recall[cls] = (pred_has & gt_has).sum() / gt_has.sum() * 100
        m["class_recall"] = class_recall

        metrics[model] = m

    return metrics


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _bar(ax, models, values, colors, title, ylim=None, ylabel="", fmt=".1f"):
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.spines[["top", "right"]].set_visible(False)
    top = ylim or (max(v for v in values if not np.isnan(v)) * 1.12 if any(not np.isnan(v) for v in values) else 1)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=7.5)


def _bar_err(ax, models, means, stds, colors, title, ylabel=""):
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.8, width=0.6,
           error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#333"})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _class_heatmap(ax, metrics, models):
    """Recall heatmap: rows=classes, cols=models."""
    classes = VLM_COMPARABLE_CLASSES
    data = np.full((len(classes), len(models)), np.nan)
    for j, model in enumerate(models):
        cr = metrics[model].get("class_recall", {})
        for i, cls in enumerate(classes):
            if cls in cr:
                data[i, j] = cr[cls]

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title("Per-class Recall (%) — when GT says class is present",
                 fontsize=11, fontweight="bold")
    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(models)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if 20 < v < 80 else "white",
                        fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999")
    plt.colorbar(im, ax=ax, shrink=0.8, label="%")


# ── Main figure ───────────────────────────────────────────────────────────────

def build_figure(metrics: dict, out: Path) -> None:
    models = list(metrics.keys())
    colors = PALETTE[: len(models)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("VLM Garbage Detector — Model Evaluation", fontsize=16, fontweight="bold", y=1.01)

    # Layout: 3 rows
    # Row 0 (3 cols): accuracy, precision, recall
    # Row 1 (3 cols): F1, inference time, VRAM
    # Row 2 (1 col spanning all): per-class recall heatmap

    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_prec = fig.add_subplot(gs[0, 1])
    ax_rec  = fig.add_subplot(gs[0, 2])
    ax_f1   = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])
    ax_vram = fig.add_subplot(gs[1, 2])
    ax_heat = fig.add_subplot(gs[2, :])

    _bar(ax_acc,  models, [metrics[m]["accuracy"]  for m in models], colors, "Accuracy (%)",  ylim=105, ylabel="%")
    _bar(ax_prec, models, [metrics[m]["precision"] for m in models], colors, "Precision (%)", ylim=105, ylabel="%")
    _bar(ax_rec,  models, [metrics[m]["recall"]    for m in models], colors, "Recall (%)",    ylim=105, ylabel="%")
    _bar(ax_f1,   models, [metrics[m]["f1"]        for m in models], colors, "F1 (%)",        ylim=105, ylabel="%")
    _bar_err(ax_time, models,
             [metrics[m]["time_mean"] for m in models],
             [metrics[m]["time_std"]  for m in models],
             colors, "Inference Time — mean ± std", ylabel="seconds")
    _bar(ax_vram, models, [metrics[m]["vram_mean"] for m in models], colors, "VRAM — mean", ylabel="MB", fmt=".0f")
    _class_heatmap(ax_heat, metrics, models)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    models = list(metrics.keys())
    col_w = max(len(m) for m in models) + 2
    sep = "─" * (col_w + 62)
    print(f"\n{sep}")
    print(f"{'Model':<{col_w}}  {'N':>5}  {'Time(s)':>8}  {'VRAM MB':>8}  {'Acc%':>6}  {'Prec%':>6}  {'Rec%':>5}  {'F1%':>5}")
    print(sep)
    for model, m in metrics.items():
        print(
            f"{model:<{col_w}}  {m['n_matched']:>5}  "
            f"{m['time_mean']:>6.2f}s  {m['vram_mean']:>7.0f}  "
            f"{m['accuracy']:>5.1f}  {m['precision']:>5.1f}  "
            f"{m['recall']:>5.1f}  {m['f1']:>5.1f}"
        )
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM garbage detector results")
    parser.add_argument("--images",  default="images",  help="Directory with images + YOLO .txt labels")
    parser.add_argument("--results", default="results", help="Directory with detections_*.csv files")
    parser.add_argument("--out",     default="results/evaluation.png", help="Output figure path")
    args = parser.parse_args()

    gt   = load_yolo_labels(Path(args.images))
    dfs  = load_results(Path(args.results))
    mets = compute_metrics(dfs, gt)
    print_summary(mets)
    build_figure(mets, Path(args.out))


if __name__ == "__main__":
    main()
