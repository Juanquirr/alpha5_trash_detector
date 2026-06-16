"""
FlowIMG evaluation configuration.

To add a new model: add one entry to MODELS dict.
All paths are relative to this file or absolute.
"""

from pathlib import Path

# Project root (one level up from this file)
ROOT = Path(__file__).parent.parent.resolve()

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Set via --dataset CLI flag in run_eval.py (no default path)

# ─── Models ───────────────────────────────────────────────────────────────────
# Key   = display name in results table
# Value = path to best.pt weights (absolute or relative to ROOT)
MODELS = {
    "YOLO26x_v10": ROOT / "runs" / "YOLO26x_v10_20260603"   / "weights" / "best.pt",
    "YOLO26x_v7":  ROOT / "runs" / "YOLO26x_v7_20260603"    / "weights" / "best.pt",
    "YOLO26x_v5_sinHN": ROOT / "runs" / "YOLO26x_v5_sinHN_20260522" / "weights" / "best.pt",
    # Add new models here:
    # "MyNewModel": ROOT / "runs" / "YOLO26x_vXX_YYYYMMDD" / "weights" / "best.pt",
}

# ─── Inference settings ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.3   # minimum detection confidence
IMGSZ         = 640     # inference image size (pixels)
DEVICE        = "0"     # "0" for first GPU, "cpu" for CPU
BATCH_SIZE    = 16      # images per predict() call

# ─── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR      = Path(__file__).parent / "results"
SAVE_PREDICTIONS = True   # save annotated sample images (good for presentation)
N_SAMPLE_IMAGES  = 20     # how many sample images to save per model
