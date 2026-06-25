"""
Batch inference over the synthetic PLOCAN bank.

Runs the detector (YOLO26x C1) with several inference methods over a folder of
synthetic images and dumps annotated images plus a per-image / per-method
detection count, ready for the results chapter (sec:producto_final).

Reuses the inference methods already implemented in the desktop prototype
(alpha5/tests/visualizer/inference_methods.py): basic, tiled, hybrid, ...

Usage:
    python infer_synthetic_bank.py \
        --model runs/YOLO26x_v10_20260603/weights/best.pt \
        --images path/to/synthetic_images \
        --out synthetic_eval_out \
        --methods basic tiled hybrid \
        --conf 0.25

Output:
    <out>/<method>/<stem>.png   annotated detections per method
    <out>/summary.csv           image, method, n_det, time_s, per-class counts
    aggregate table printed to stdout
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# --- make the prototype's inference methods importable (no package layout) ---
VIS_DIR = Path(__file__).resolve().parent / "alpha5" / "tests" / "visualizer"
sys.path.insert(0, str(VIS_DIR))

from inference_methods import METHODS, get_method  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_images(images_dir: Path, pattern: str, skip_debug: bool):
    imgs = []
    for p in sorted(images_dir.rglob(pattern)):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if skip_debug and "_debug" in p.stem:
            continue
        imgs.append(p)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/YOLO26x_v10_20260603/weights/best.pt",
                    help="Path to the detector checkpoint (.pt)")
    ap.add_argument("--images", required=True, help="Folder with synthetic images")
    ap.add_argument("--out", default="synthetic_eval_out", help="Output folder")
    ap.add_argument("--methods", nargs="+", default=["basic", "tiled", "hybrid"],
                    choices=list(METHODS.keys()))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--trash-id", type=int, default=5,
                    help="Class id of 'trash' in the model (V10 = 5)")
    ap.add_argument("--pattern", default="*", help="Glob pattern for images")
    ap.add_argument("--keep-debug", action="store_true",
                    help="Do not skip files whose name contains '_debug'")
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        sys.exit(f"Images folder not found: {images_dir}")

    images = discover_images(images_dir, args.pattern, skip_debug=not args.keep_debug)
    if not images:
        sys.exit(f"No images found in {images_dir} (pattern={args.pattern})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in args.methods:
        (out_dir / m).mkdir(exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    names = model.names  # id -> class name

    rows = []
    agg = defaultdict(lambda: {"n_images": 0, "n_det": 0, "time": 0.0})

    print(f"Running {len(args.methods)} method(s) over {len(images)} image(s)...\n")
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue

        for method_name in args.methods:
            method = get_method(method_name)
            params = dict(method.default_params)
            params["conf"] = args.conf
            params["trash_class_id"] = args.trash_id

            t0 = time.time()
            result = method.run(image.copy(), model, params)
            elapsed = time.time() - t0

            cls_counts = Counter(int(c) for c in result.classes)
            cls_str = ";".join(f"{names.get(c, c)}:{n}" for c, n in sorted(cls_counts.items()))

            out_path = out_dir / method_name / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), result.image)

            rows.append({
                "image": img_path.name,
                "method": method_name,
                "n_det": int(result.num_detections),
                "time_s": round(elapsed, 3),
                "classes": cls_str,
            })
            a = agg[method_name]
            a["n_images"] += 1
            a["n_det"] += int(result.num_detections)
            a["time"] += elapsed

        print(f"  {img_path.name} done")

    # --- write CSV ---
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "method", "n_det", "time_s", "classes"])
        w.writeheader()
        w.writerows(rows)

    # --- aggregate table ---
    print("\n================ AGGREGATE ================")
    print(f"{'method':<12}{'imgs':>6}{'total_det':>11}{'det/img':>10}{'avg_s':>9}")
    for m in args.methods:
        a = agg[m]
        n = a["n_images"] or 1
        print(f"{m:<12}{a['n_images']:>6}{a['n_det']:>11}{a['n_det']/n:>10.2f}{a['time']/n:>9.3f}")
    print("===========================================")
    print(f"\nPer-image CSV : {csv_path}")
    print(f"Annotated imgs: {out_dir}/<method>/")


if __name__ == "__main__":
    main()
