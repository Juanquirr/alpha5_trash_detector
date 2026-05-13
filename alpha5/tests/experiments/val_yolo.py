import argparse
from pathlib import Path
import yaml
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full YOLO validation + optional val predictions concat export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("data", type=str, help="Path to data.yaml.")
    p.add_argument("model", type=str, help="Path to YOLO weights (.pt).")

    p.add_argument("--project", type=str, default="full_validation_run", help="Ultralytics project output dir.")
    p.add_argument("--name", type=str, default="val_full", help="Ultralytics run name under project.")

    p.add_argument("--imgsz", type=int, default=640, help="Validation/prediction image size.")
    p.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.35, help="IoU threshold.")

    p.add_argument("--device", type=str, default=None, help="Device (e.g. cpu, cuda, cuda:0).")

    p.add_argument("--save_json", action="store_true", help="Save COCO-style JSON in validation.")
    p.add_argument("--save_hybrid", action="store_true", help="Save hybrid labels in validation.")
    p.add_argument("--plots", action="store_true", help="Save validation plots (incl. confusion matrix).")

    p.add_argument("--per_class_csv", action="store_true", help="Export per-class metrics CSV (P, R, F1, mAP50, mAP50-95, contrib%).")
    p.add_argument("--plot_classes", action="store_true", help="Save bar chart PNG with per-class mAP50-95 and contribution.")
    p.add_argument("--predict_val", action="store_true", help="Run prediction over val images after validation.")
    p.add_argument("--val_images", type=str, default=None, help="Override val images dir (optional).")

    p.add_argument("--concat", action="store_true", help="Save side-by-side (original | prediction) images.")
    p.add_argument("--concat_dirname", type=str, default="predictions_val_concat", help="Concat output folder name.")

    p.add_argument("--recursive", action="store_true", help="Recursively search images in val_images.")
    return p.parse_args()

def resolve_val_images_dir(data_yaml: Path) -> Path:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    val_entry = cfg.get("val", None)
    base = data_yaml.parent

    # Common YOLO case: val is a relative path like "val/images"
    if isinstance(val_entry, str):
        p = Path(val_entry)
        return p if p.is_absolute() else (base / p)

    # Fallback to the typical structure
    return base / "val" / "images"

def list_images(dir_path: Path, recursive: bool) -> list[Path]:
    if not dir_path.exists():
        return []
    it = dir_path.rglob("*") if recursive else dir_path.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])

def resize_to_same_height(img_a, img_b):
    if img_a is None or img_b is None:
        return None, None
    if img_a.shape[0] == img_b.shape[0]:
        return img_a, img_b
    h = min(img_a.shape[0], img_b.shape[0])
    img_a = cv2.resize(img_a, (int(img_a.shape[1] * h / img_a.shape[0]), h))
    img_b = cv2.resize(img_b, (int(img_b.shape[1] * h / img_b.shape[0]), h))
    return img_a, img_b

def run_validation(model: YOLO, args: argparse.Namespace):
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        plots=args.plots,
    )
    return metrics

def _build_per_class_rows(model: YOLO, metrics) -> list[dict]:
    """
    Align per-class arrays from metrics.box with model.names.
    metrics.box arrays (p, r, ap50, ap) are indexed by position
    in ap_class_index, NOT by raw class ID.
    """
    ap_class_index = list(metrics.box.ap_class_index)  # [class_id, ...]
    p_arr   = list(metrics.box.p)    # precision per position
    r_arr   = list(metrics.box.r)    # recall per position
    ap50arr = list(metrics.box.ap50) # AP@0.50 per position
    ap_arr  = list(metrics.box.ap)   # AP@0.50-95 per position

    # Build lookup: class_id -> index in the above arrays
    idx_map = {int(cid): i for i, cid in enumerate(ap_class_index)}

    # Total AP pool = sum of per-class AP@0.50-95 (classes absent in val contribute 0).
    # contrib_% = ap_class / total_ap * 100  → sums to 100 %, equal share = 100/N %.
    total_ap = sum(float(v) for v in ap_arr) or 1e-9

    rows = []
    for class_id, class_name in sorted(model.names.items()):
        i = idx_map.get(int(class_id))
        if i is not None:
            p    = float(p_arr[i])
            r    = float(r_arr[i])
            ap50 = float(ap50arr[i])
            ap   = float(ap_arr[i])
        else:
            p = r = ap50 = ap = 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        rows.append({
            "class_id":   int(class_id),
            "class_name": str(class_name),
            "Precision":  round(p,    4),
            "Recall":     round(r,    4),
            "F1":         round(f1,   4),
            "mAP50":      round(ap50, 4),
            "mAP50-95":   round(ap,   4),
            "contrib_%":  round(ap / total_ap * 100, 2),
        })
    return rows


def print_global_metrics(metrics):
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    print("\n===== GLOBAL METRICS =====")
    print(f"mAP@0.50:     {float(metrics.box.map50):.4f}")
    print(f"mAP@0.50-95:  {float(metrics.box.map):.4f}")
    print(f"mAP@0.75:     {float(metrics.box.map75):.4f}")
    print(f"Precision:    {mp:.4f}")
    print(f"Recall:       {mr:.4f}")
    print(f"F1-Score:     {f1:.4f}")


def print_per_class_table(model: YOLO, metrics):
    rows = _build_per_class_rows(model, metrics)
    header = f"\n{'ID':>3}  {'Class':<20}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mAP50':>7}  {'mAP50-95':>9}  {'Contrib%':>9}"
    sep    = "-" * len(header)
    print("\n===== PER-CLASS METRICS =====")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['class_id']:>3}  {r['class_name']:<20}  "
            f"{r['Precision']:>6.4f}  {r['Recall']:>6.4f}  {r['F1']:>6.4f}  "
            f"{r['mAP50']:>7.4f}  {r['mAP50-95']:>9.4f}  {r['contrib_%']:>8.2f}%"
        )
    print(sep)


def export_per_class_csv(model: YOLO, metrics, out_dir: Path):
    rows = _build_per_class_rows(model, metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_per_class.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Per-class CSV saved: {out_path}")


def plot_per_class(model: YOLO, metrics, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skip --plot_classes")
        return

    rows = _build_per_class_rows(model, metrics)
    names    = [r["class_name"]  for r in rows]
    map5095  = [r["mAP50-95"]   for r in rows]
    map50    = [r["mAP50"]       for r in rows]
    contrib  = [r["contrib_%"]   for r in rows]
    f1       = [r["F1"]          for r in rows]

    x = np.arange(len(names))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(names) * 1.2), 9))
    fig.suptitle("Per-class metrics", fontsize=14, fontweight="bold")

    # Top: mAP50, mAP50-95, F1
    ax1.bar(x - w,     map50,   w, label="mAP@0.50",   color="#4C9BE8")
    ax1.bar(x,         map5095, w, label="mAP@0.50-95", color="#E84C4C")
    ax1.bar(x + w,     f1,      w, label="F1",          color="#4CE89B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, map5095):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    # Bottom: contribution % to global mAP50-95
    bars = ax2.bar(x, contrib, color="#E8A94C")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.axhline(100 / len(names), color="red", linestyle="--",
                linewidth=1, label=f"Equal share ({100/len(names):.1f}%)")
    ax2.set_ylabel("Contribution to global mAP50-95 (%)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, contrib):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_per_class.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Per-class chart saved: {out_path}")

def predict_and_optional_concat(model: YOLO, args: argparse.Namespace, val_images_dir: Path, run_dir: Path):
    images = list_images(val_images_dir, recursive=args.recursive)
    if not images:
        print(f"No val images found at: {val_images_dir}")
        return 0

    concat_dir = run_dir / args.concat_dirname
    if args.concat:
        concat_dir.mkdir(parents=True, exist_ok=True)

    results_iter = model.predict(
        source=[str(p) for p in images],
        stream=True,                 # generator, lower memory
        save_txt=True,
        save_conf=True,
        save=False,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name="predictions_val",
    )

    with tqdm(total=len(images), desc="Predicting val", unit="img") as pbar:
        written = 0
        for r in results_iter:
            if args.concat:
                img_path = Path(r.path)
                original = cv2.imread(str(img_path))
                pred_img = r.plot()  # numpy BGR
                original, pred_img = resize_to_same_height(original, pred_img)
                if original is not None and pred_img is not None:
                    out = cv2.hconcat([original, pred_img])
                    cv2.imwrite(str(concat_dir / f"{img_path.stem}_concat.jpg"), out)
                    written += 1

            pbar.update(1)

    return written

def main():
    args = parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    model = YOLO(str(model_path))

    metrics = run_validation(model, args)
    print_global_metrics(metrics)
    print_per_class_table(model, metrics)

    run_dir = Path(args.project) / args.name

    if args.per_class_csv:
        export_per_class_csv(model, metrics, out_dir=run_dir)

    if args.plot_classes:
        plot_per_class(model, metrics, out_dir=run_dir)

    if args.predict_val:
        val_images_dir = Path(args.val_images) if args.val_images else resolve_val_images_dir(data_yaml)
        written = predict_and_optional_concat(model, args, val_images_dir, run_dir=run_dir)
        if args.concat:
            print(f"Concat images written: {written}")

    if args.plots:
        print(f"Plots/confusion matrix should be under: {run_dir}")  # confusion matrix is generated by plots=True

if __name__ == "__main__":
    main()
