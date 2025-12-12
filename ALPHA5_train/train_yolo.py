import torch
import argparse
import os
import yaml
from ultralytics import YOLO


def build_arguments() -> argparse.Namespace:
    """
    Builds the command-line argument parser for YOLO training.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics YOLO model."
    )

    # Required arguments
    parser.add_argument(
        "data",
        type=str,
        help="Path to the dataset data.yaml file",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Ultralytics YOLO model spec (e.g., yolo11x.pt or /path/to/weights.pt)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for AutoBatch)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, cuda:0, ...)")

    # Output arguments
    parser.add_argument(
        "--project",
        type=str,
        default="/ultralytics/plocania/runs/detect/train",
        help="Project directory where results are saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (subfolder under project)",
    )

    # Optimizer / hyperparameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Path to a YAML hyperparameters file (optional)",
    )

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Print CUDA / PyTorch info")

    return parser.parse_args()


def print_cuda_info() -> None:
    """Prints PyTorch and CUDA information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 50)


def train_yolo(args: argparse.Namespace) -> None:
    best_map50 = 0.0
    patience_counter = 0
    epoch_log = []

    def on_fit_epoch_end(trainer):
        nonlocal best_map50, patience_counter, epoch_log

        current_map50 = trainer.metrics.get("metrics/mAP50(B)", None)
        if current_map50 is None:
            return

        if current_map50 > best_map50:
            best_map50 = current_map50
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_1based = trainer.epoch + 1
        print(
            f"Epoch {epoch_1based}: mAP50 = {current_map50:.4f}, "
            f"Best = {best_map50:.4f}, Patience = {patience_counter}/{args.patience}"
        )

        epoch_log.append(
            {
                "epoch": epoch_1based,
                "map50": float(current_map50),
                "best_map50": float(best_map50),
                "patience_counter": int(patience_counter),
            }
        )

    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"CUDA not available, using CPU instead of {args.device}")

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Dataset: {args.data}")
    print(f"Results will be saved to: {args.project}")
    print("-" * 50)

    hparams = {}
    if args.hyperparams:
        with open(args.hyperparams, "r") as f:
            hparams = yaml.safe_load(f) or {}

        if "close_mosaic" in hparams and hparams["close_mosaic"] is not None:
            hparams["close_mosaic"] = int(hparams["close_mosaic"])

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        **hparams,
    )

    print(f"Total trained epochs logged by callback: {len(epoch_log)}")


if __name__ == "__main__":
    args = build_arguments()

    if args.verbose:
        print_cuda_info()

    if not os.path.exists(args.data):
        print(f"Error: data.yaml not found at: {args.data}")
        raise SystemExit(1)

    train_yolo(args)
