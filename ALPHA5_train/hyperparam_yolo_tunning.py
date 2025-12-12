import argparse
import os
import yaml
import torch
from ultralytics import YOLO


RESERVED_TUNE_KEYS = {
    "data", "model", "epochs", "iterations", "batch", "imgsz", "patience",
    "device", "name", "project", "resume"
}


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultralytics YOLO hyperparameter tuning (model.tune)")

    p.add_argument("data", type=str, help="Path to dataset data.yaml")
    p.add_argument("model", type=str, help="Model spec or weights path (e.g., yolo11x.pt)")

    p.add_argument("epochs", type=int, help="Epochs per tuning iteration")
    p.add_argument("iterations", type=int, help="Number of tuning iterations")

    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for AutoBatch)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--patience", type=int, help="Early-stopping patience")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, cuda:0, ...)")

    p.add_argument("--name", type=str, default="tune_exp", help="Tuning run name")
    p.add_argument("--project", type=str, default=None, help="Output project directory (optional)")
    p.add_argument("--resume", action="store_true", help="Resume an existing tuning run with same name/project")

    p.add_argument(
        "--tune_kwargs",
        type=str,
        default=None,
        help="Optional YAML with extra starting kwargs for model.tune() (e.g., lr0, lrf, momentum)",
    )

    p.add_argument("--verbose", action="store_true", help="Print CUDA / PyTorch info")
    return p.parse_args()


def load_optional_tune_kwargs(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    d = {k: v for k, v in d.items() if k not in RESERVED_TUNE_KEYS}

    if "close_mosaic" in d and d["close_mosaic"] is not None:
        d["close_mosaic"] = int(float(d["close_mosaic"]))

    return d


def main():
    args = build_args()

    if args.verbose:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.data):
        raise SystemExit(f"Error: data.yaml not found at: {args.data}")

    tune_extra = load_optional_tune_kwargs(args.tune_kwargs)

    model = YOLO(args.model)

    tune_call_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device if torch.cuda.is_available() else "cpu",
        name=args.name,
        resume=args.resume,
    )
    if args.project:
        tune_call_kwargs["project"] = args.project

    for k, v in tune_extra.items():
        tune_call_kwargs.setdefault(k, v)

    model.tune(**tune_call_kwargs)

    print("\nTuning finished. Check the run directory under runs/detect/tune/ for best hyperparameters.")

if __name__ == "__main__":
    main()
