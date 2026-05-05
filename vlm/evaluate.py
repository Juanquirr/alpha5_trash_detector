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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Class maps ────────────────────────────────────────────────────────────────

YOLO_ID_TO_CLASS = {
    0: "plastic bottle",
    1: "glass",
    2: "can",
    3: "plastic bag",
    4: "metal scrap",
    5: "plastic wrapper",
    6: "trash pile",
    7: "trash",
}

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
    rows = []
    for txt in sorted(images_dir.glob("*.txt")):
        image_name = txt.with_suffix(".jpg").name
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

def compute_metrics(dfs: dict[str, pd.DataFrame], gt: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (per-model metrics dict, class_gt_counts dict)."""
    metrics = {}

    # Count GT instances per class across all images
    class_gt_counts = {cls: int(gt["gt_classes"].apply(lambda s: cls in s).sum())
                       for cls in VLM_COMPARABLE_CLASSES}

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
            m["tp"] = m["fp"] = m["fn"] = m["tn"] = 0
            m["class_recall"] = {}
            metrics[model] = m
            continue

        pred = merged["garbage_detected"].astype(bool)
        true = merged["gt_detected"].astype(bool)
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        tn = int((~pred & ~true).sum())

        m["tp"], m["fp"], m["fn"], m["tn"] = tp, fp, fn, tn
        m["accuracy"]  = (tp + tn) / n * 100
        m["precision"] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        m["recall"]    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        m["f1"]        = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0.0

        def _parse_classes(s: str) -> frozenset:
            return frozenset(c.strip().lower() for c in s.split(",") if c.strip())

        pred_class_sets = merged["classes_detected"].fillna("").apply(_parse_classes)

        class_recall = {}
        for cls in VLM_COMPARABLE_CLASSES:
            gt_has = merged["gt_classes"].apply(lambda s: cls in s)
            if gt_has.sum() == 0:
                continue
            pred_has = pred_class_sets.apply(lambda s: cls in s)
            class_recall[cls] = (pred_has & gt_has).sum() / gt_has.sum() * 100
        m["class_recall"] = class_recall

        metrics[model] = m

    return metrics, class_gt_counts


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _section_header_ax(ax, text, color="#2c3e50"):
    """Fill a spanning axes row as a colored section banner (no overlay)."""
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _bar(ax, models, values, colors, title, subtitle,
         ylim=None, ylabel="", fmt=".1f", higher_better=True,
         ref_lines=None):
    """Bar chart with descriptive subtitle, reference lines, and direction hint."""
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6, zorder=3)

    # Reference lines (e.g. 50% baseline, 80% target)
    if ref_lines:
        for val, label, style in ref_lines:
            ax.axhline(val, color="#888", linewidth=0.8, linestyle=style, zorder=2)
            ax.text(len(models) - 0.5, val + 1.5, label,
                    ha="right", va="bottom", fontsize=6.5, color="#666")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(subtitle, fontsize=7.5, color="#555", labelpad=6, wrap=True)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    arrow = "↑ higher better" if higher_better else "↓ lower better"
    ax.text(1.0, 1.02, arrow, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#888", style="italic")

    top = ylim or (max((v for v in values if not np.isnan(v)), default=1) * 1.12)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")


def _bar_err(ax, models, means, stds, colors, title, subtitle, ylabel=""):
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.8, width=0.6,
           error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#333"}, zorder=3)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(subtitle, fontsize=7.5, color="#555", labelpad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.0, 1.02, "↓ lower better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#888", style="italic")

    for xi, (mean, std) in enumerate(zip(means, stds)):
        if not np.isnan(mean):
            ax.text(xi, mean + (std or 0) + max(means) * 0.02,
                    f"{mean:.2f}s", ha="center", va="bottom", fontsize=7.5, fontweight="bold")


def _class_heatmap(ax, metrics, models, class_gt_counts):
    classes = VLM_COMPARABLE_CLASSES
    data = np.full((len(classes), len(models)), np.nan)
    for j, model in enumerate(models):
        cr = metrics[model].get("class_recall", {})
        for i, cls in enumerate(classes):
            if cls in cr:
                data[i, j] = cr[cls]

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(len(classes)))

    # Row labels with GT counts
    ylabels = [
        f"{cls}  (GT: {class_gt_counts.get(cls, 0)} imgs)"
        for cls in classes
    ]
    ax.set_yticklabels(ylabels, fontsize=8.5)

    ax.set_title(
        "Per-class Recall  —  of all images where GT contains a class, "
        "what % did the model mention it?  (red = missed, green = detected)",
        fontsize=10, fontweight="bold", pad=8
    )

    # Cell annotations
    for i in range(len(classes)):
        for j in range(len(models)):
            v = data[i, j]
            if not np.isnan(v):
                text_color = "black" if 20 < v < 80 else "white"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#aaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("Recall (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _dataset_info_box(fig, gt: pd.DataFrame, metrics: dict, y: float):
    """Top info strip: dataset stats + per-model image count."""
    n_total   = len(gt)
    n_garbage = int(gt["gt_detected"].sum())
    n_clean   = n_total - n_garbage

    class_counts = {cls: int(gt["gt_classes"].apply(lambda s: cls in s).sum())
                    for cls in VLM_COMPARABLE_CLASSES}
    top_classes = sorted(class_counts.items(), key=lambda x: -x[1])

    dataset_str = (
        f"Dataset: {n_total} images  |  "
        f"{n_garbage} with garbage ({n_garbage/n_total*100:.0f}%)  |  "
        f"{n_clean} clean ({n_clean/n_total*100:.0f}%)"
    )
    class_str = "  ·  ".join(f"{cls}: {n}" for cls, n in top_classes if n > 0)

    fig.text(0.5, y + 0.012, dataset_str, ha="center", va="center",
             fontsize=9, fontweight="bold", color="#2c3e50")
    fig.text(0.5, y - 0.005, f"Class distribution  →  {class_str}",
             ha="center", va="center", fontsize=7.5, color="#555")


# ── Main figure ───────────────────────────────────────────────────────────────

def build_figure(metrics: dict, gt: pd.DataFrame, class_gt_counts: dict, out: Path) -> None:
    models = list(metrics.keys())
    colors = PALETTE[: len(models)]

    fig = plt.figure(figsize=(22, 21))
    fig.patch.set_facecolor("#f8f9fa")

    # 5-row grid: [header1, charts1, header2, charts2, heatmap]
    # Info strip lives above the GridSpec via fig.text
    gs = GridSpec(5, 3, figure=fig,
                  height_ratios=[0.07, 1, 0.07, 1, 1.5],
                  hspace=0.45, wspace=0.33,
                  top=0.89, bottom=0.04)

    fig.suptitle("VLM Garbage Detector — Model Evaluation Report",
                 fontsize=17, fontweight="bold", y=0.975, color="#1a1a2e")

    # Dataset info strip (above GridSpec)
    _dataset_info_box(fig, gt, metrics, y=0.925)

    # Section header axes — proper rows, no overlay
    _section_header_ax(
        fig.add_subplot(gs[0, :]),
        "▌ CLASSIFICATION QUALITY  —  binary detection: garbage vs clean",
    )
    _section_header_ax(
        fig.add_subplot(gs[2, :]),
        "▌ EFFICIENCY  —  compute cost per image",
    )

    # Row 1: classification quality
    ax_acc  = fig.add_subplot(gs[1, 0])
    ax_prec = fig.add_subplot(gs[1, 1])
    ax_rec  = fig.add_subplot(gs[1, 2])

    ref_pct = [
        (50, "50% baseline", "--"),
        (80, "80% target",   ":"),
    ]

    _bar(ax_acc, models, [metrics[m]["accuracy"] for m in models], colors,
         "Accuracy",
         "(TP + TN) / total  —  % of all images correctly labelled as garbage or clean",
         ylim=105, ylabel="%", ref_lines=ref_pct)

    _bar(ax_prec, models, [metrics[m]["precision"] for m in models], colors,
         "Precision",
         "TP / (TP + FP)  —  when the model flags garbage, how often is it right?\n"
         "Low precision = many false alarms on clean images",
         ylim=105, ylabel="%", ref_lines=ref_pct)

    _bar(ax_rec, models, [metrics[m]["recall"] for m in models], colors,
         "Recall  (Sensitivity)",
         "TP / (TP + FN)  —  of all images that contain garbage, how many did the model catch?\n"
         "Low recall = garbage being missed",
         ylim=105, ylabel="%", ref_lines=ref_pct)

    # Row 2: efficiency + F1
    ax_f1   = fig.add_subplot(gs[3, 0])
    ax_time = fig.add_subplot(gs[3, 1])
    ax_vram = fig.add_subplot(gs[3, 2])

    _bar(ax_f1, models, [metrics[m]["f1"] for m in models], colors,
         "F1 Score",
         "2·TP / (2·TP + FP + FN)  —  harmonic mean of precision & recall.\n"
         "Best single-number summary when dataset is imbalanced",
         ylim=105, ylabel="%", ref_lines=ref_pct)

    _bar_err(ax_time, models,
             [metrics[m]["time_mean"] for m in models],
             [metrics[m]["time_std"]  for m in models],
             colors,
             "Inference Time",
             "Seconds per image (mean ± std).  Bars show average; error bars show variability.\n"
             "Low mean + low std = fast and consistent",
             ylabel="seconds")

    _bar(ax_vram, models,
         [metrics[m]["vram_mean"] for m in models], colors,
         "GPU Memory (VRAM)",
         "Peak MB allocated during inference (mean across images).\n"
         "Determines minimum GPU hardware needed for deployment",
         ylabel="MB", fmt=".0f", higher_better=False)

    # Row 3: per-class heatmap (spans all 3 columns)
    ax_heat = fig.add_subplot(gs[4, :])
    _class_heatmap(ax_heat, metrics, models, class_gt_counts)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    models = list(metrics.keys())
    col_w = max(len(m) for m in models) + 2
    sep = "─" * (col_w + 65)
    print(f"\n{sep}")
    print(f"{'Model':<{col_w}}  {'N':>5}  {'Time(s)':>8}  {'VRAM MB':>8}  "
          f"{'Acc%':>6}  {'Prec%':>6}  {'Rec%':>5}  {'F1%':>5}")
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

    gt                   = load_yolo_labels(Path(args.images))
    dfs                  = load_results(Path(args.results))
    mets, class_gt_counts = compute_metrics(dfs, gt)
    print_summary(mets)
    build_figure(mets, gt, class_gt_counts, Path(args.out))


if __name__ == "__main__":
    main()
