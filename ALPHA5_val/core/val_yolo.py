import argparse
from pathlib import Path
import yaml
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full YOLO validation + optional val predictions concat export.")

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

    p.add_argument("--per_class_csv", action="store_true", help="Export per-class mAP50-95 CSV.")
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

def print_global_metrics(metrics):
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    print("===== GLOBAL METRICS =====")
    print(f"mAP@0.50:     {float(metrics.box.map50):.4f}")
    print(f"mAP@0.50-95:  {float(metrics.box.map):.4f}")
    print(f"mAP@0.75:     {float(metrics.box.map75):.4f}")
    print(f"Precision:    {mp:.4f}")
    print(f"Recall:       {mr:.4f}")
    print(f"F1-Score:     {f1:.4f}")

def export_per_class_csv(model: YOLO, metrics, out_dir: Path):
    maps = metrics.box.maps  # mAP50-95 per class
    rows = []
    for class_id, class_name in model.names.items():
        v = float(maps[class_id]) if class_id < len(maps) else 0.0
        rows.append({"class_id": int(class_id), "class_name": str(class_name), "mAP50-95": v})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_per_class.csv", index=False)

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

    run_dir = Path(args.project) / args.name

    if args.per_class_csv:
        export_per_class_csv(model, metrics, out_dir=run_dir)

    if args.predict_val:
        val_images_dir = Path(args.val_images) if args.val_images else resolve_val_images_dir(data_yaml)
        written = predict_and_optional_concat(model, args, val_images_dir, run_dir=run_dir)
        if args.concat:
            print(f"Concat images written: {written}")

    if args.plots:
        print(f"Plots/confusion matrix should be under: {run_dir}")  # confusion matrix is generated by plots=True

if __name__ == "__main__":
    main()
