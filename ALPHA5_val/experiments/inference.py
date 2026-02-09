import argparse
import os
import time
from pathlib import Path
import psutil
import torch
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(source: Path):
    if source.is_file():
        if source.suffix.lower() in IMG_EXTS:
            return [source]
        return []
    if source.is_dir():
        return [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return []

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Ultralytics YOLO inference and log time/memory per image.")
    p.add_argument("source", type=str, help="Input image path or directory containing images.")
    p.add_argument("model", type=str, help="Path to YOLO .pt weights.")
    p.add_argument("out_dir", type=str, help="Output directory for annotated images.")
    p.add_argument("--device", type=str, default="cuda", help="Device (e.g., cpu, cuda, cuda:0).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    return p.parse_args()

def main():
    args = build_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    process = psutil.Process(os.getpid())

    images = list_images(source)
    if not images:
        raise SystemExit(f"No images found in: {source}")

    for img_path in images:
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.monotonic()

        results = model.predict(source=str(img_path), device=device, imgsz=args.imgsz, conf=args.conf)

        t1 = time.monotonic()
        mem_after = process.memory_info().rss / (1024 * 1024)

        elapsed = t1 - t0
        mem_delta = mem_after - mem_before

        print(f"Image: {img_path.name}")
        print(f"  - Processing time: {elapsed:.4f} seconds")
        print(f"  - RSS memory delta: {mem_delta:.2f} MB")

        out_path = out_dir / f"detect_{img_path.name}"
        results[0].save(filename=str(out_path))

    print("Done.")

if __name__ == "__main__":
    main()
