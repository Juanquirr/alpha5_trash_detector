"""
Bubble plot (4D) comparing VLM models on V10 POPE evaluation.

Axes:
  X — Median inference time (seconds per question)
  Y — F1 score (averaged across 3 POPE levels, valid responses only)
  Bubble size — Model parameters (billions)
  Bubble color — |Yes-ratio − 0.5| (calibration deviation; lower = better)

Extras:
  - Dashed arrows from pre-LoRA to post-LoRA per model
  - Pareto frontier on post-LoRA points (lower time + higher F1 = dominant)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
import os, glob

BASE = Path(__file__).parent / "pope_results_v10"

MODEL_META = {
    "qwen_2b":       {"label": "Qwen3-VL-2B",        "params": 2.7},
    "qwen_vl":       {"label": "Qwen2.5-VL-3B",      "params": 3.8},
    "smolvlm":       {"label": "SmolVLM-1B",          "params": 1.0},
    "smolvlm_500m":  {"label": "SmolVLM-500M",        "params": 0.5},
    "llava":         {"label": "LLaVA-OneVision-0.5B", "params": 0.5},
}

TIMEOUT_S = 15.0


def compute_metrics(dir_path: Path) -> dict | None:
    csvs = sorted(dir_path.glob("*.csv"))
    if not csvs:
        return None

    dfs = [pd.read_csv(f) for f in csvs]
    all_df = pd.concat(dfs, ignore_index=True)

    resp = all_df["response"].astype(str).str.strip().str.lower().str.rstrip(".")
    valid = resp.isin(["yes", "no"])

    vdf = all_df[valid].copy()
    if len(vdf) == 0:
        return None

    vdf["resp_norm"] = vdf["response"].astype(str).str.strip().str.lower().str.rstrip(".")
    vdf["label_norm"] = vdf["label"].astype(str).str.strip().str.lower().str.rstrip(".")

    tp = ((vdf["resp_norm"] == "yes") & (vdf["label_norm"] == "yes")).sum()
    fp = ((vdf["resp_norm"] == "yes") & (vdf["label_norm"] == "no")).sum()
    fn = ((vdf["resp_norm"] == "no") & (vdf["label_norm"] == "yes")).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    yes_ratio = (vdf["resp_norm"] == "yes").mean()
    valid_rate = valid.sum() / len(all_df)
    median_inf = all_df["inference_s"].median()

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "yes_ratio": yes_ratio,
        "valid_rate": valid_rate,
        "median_inf_s": median_inf,
        "n_total": len(all_df),
    }


def gather_data():
    rows = []
    for model_key, meta in MODEL_META.items():
        for stage in ["pre", "post"]:
            dir_name = f"{model_key}_{stage}"
            dir_path = BASE / dir_name
            if not dir_path.is_dir():
                print(f"  [skip] {dir_path} not found")
                continue

            m = compute_metrics(dir_path)
            if m is None:
                print(f"  [skip] {dir_name}: no valid data")
                continue

            rows.append({
                "model": meta["label"],
                "model_key": model_key,
                "stage": stage,
                "params_b": meta["params"],
                **m,
            })

    return pd.DataFrame(rows)


def pareto_frontier(df):
    """Return indices of Pareto-optimal points (minimize X, maximize Y)."""
    pts = df[["median_inf_s", "f1"]].values
    is_pareto = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    is_pareto[i] = False
                    break
    return df.index[is_pareto]


def plot(df):
    fig, ax = plt.subplots(figsize=(10, 7))

    calib_dev = (df["yes_ratio"] - 0.5).abs()
    norm = mcolors.Normalize(vmin=0, vmax=calib_dev.max() * 1.1)
    cmap = plt.cm.RdYlGn_r

    size_scale = 350
    sizes = df["params_b"] * size_scale

    sc = ax.scatter(
        df["median_inf_s"],
        df["f1"],
        s=sizes,
        c=calib_dev,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.85,
        zorder=5,
    )

    # Pareto frontier on post-LoRA only
    post = df[df["stage"] == "post"].copy().reset_index(drop=True)
    pareto_idx = pareto_frontier(post)
    pareto_pts = post.loc[pareto_idx].sort_values("median_inf_s")

    if len(pareto_pts) > 1:
        ax.plot(
            pareto_pts["median_inf_s"],
            pareto_pts["f1"],
            "r--",
            linewidth=1.5,
            alpha=0.7,
            zorder=4,
            label="Pareto frontier (post-LoRA)",
        )

    for idx in pareto_idx:
        row = post.loc[idx]
        ax.scatter(
            row["median_inf_s"],
            row["f1"],
            s=row["params_b"] * size_scale + 120,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=6,
        )

    # Arrows pre → post per model
    for model_key in df["model_key"].unique():
        sub = df[df["model_key"] == model_key]
        if len(sub) < 2:
            continue
        pre_row = sub[sub["stage"] == "pre"].iloc[0]
        post_row = sub[sub["stage"] == "post"].iloc[0]
        ax.annotate(
            "",
            xy=(post_row["median_inf_s"], post_row["f1"]),
            xytext=(pre_row["median_inf_s"], pre_row["f1"]),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                lw=1.2,
                linestyle="--",
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=3,
        )

    # Labels — manual offsets to avoid overlaps
    LABEL_OFFSETS = {
        ("qwen_2b", "post"):      (10, 10),
        ("qwen_2b", "pre"):       (10, -14),
        ("qwen_vl", "post"):      (-12, -18),
        ("qwen_vl", "pre"):       (-12, 8),
        ("smolvlm", "post"):      (10, 6),
        ("smolvlm", "pre"):       (10, -12),
        ("smolvlm_500m", "post"): (10, 6),
        ("smolvlm_500m", "pre"):  (10, -12),
        ("llava", "post"):        (10, 6),
        ("llava", "pre"):         (10, -12),
    }

    for _, row in df.iterrows():
        suffix = "" if row["stage"] == "post" else " (pre)"
        label = f"{row['model']}{suffix}"
        offset = LABEL_OFFSETS.get((row["model_key"], row["stage"]), (8, 6))
        fontweight = "bold" if row["stage"] == "post" else "normal"
        fontsize = 8.5 if row["stage"] == "post" else 7.5
        ha = "right" if offset[0] < 0 else "left"
        ax.annotate(
            label,
            (row["median_inf_s"], row["f1"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            zorder=7,
        )

    ax.set_xlabel("Median inference time (s/question)", fontsize=11)
    ax.set_ylabel("F1 (mean across POPE levels)", fontsize=11)
    ax.set_title("VLM comparison on V10 — POPE evaluation", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("|Yes-ratio − 0.5|  (calibration deviation)", fontsize=9)

    # Size legend
    param_vals = sorted(df["params_b"].unique())
    legend_bubbles = []
    for p in param_vals:
        legend_bubbles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=np.sqrt(p * size_scale) / 2.5,
                label=f"{p:.1f}B params",
            )
        )
    legend_bubbles.append(
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Pareto frontier")
    )
    legend_bubbles.append(
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2,
            markersize=10,
            label="Pareto-optimal",
        )
    )
    ax.legend(handles=legend_bubbles, loc="lower left", fontsize=8, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = Path(__file__).parent / "bubbleplot_vlm_v10.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    df = gather_data()
    print(df[["model", "stage", "f1", "median_inf_s", "yes_ratio", "valid_rate", "params_b"]].to_string(index=False))
    print()
    plot(df)
