"""
Usage:
    python run.py --model smolvlm --images images/
    python run.py --model moondream --images images/foto1.jpg
    python run.py --model all --images images/

Venv reference:
    .transformers-5.X-venv  (transformers 5.x) : smolvlm, qwen_vl, videollama3
    .transformers-4.46-venv (transformers 4.46): moondream, llava, blip2, instructblip,
                                                 clip, paligemma, idefics, mplug_owl3
"""
import argparse
from pathlib import Path

from models import REGISTRY, VENV
from results import append_row

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def run_model(model_key: str, images: list[Path]) -> None:
    cls = REGISTRY[model_key]
    vlm = cls()
    print(f"\n[{model_key}] Loading model...  (venv: {VENV[model_key]})")
    vlm.load()
    print(f"[{model_key}] Model ready. Processing {len(images)} image(s)...")

    for img in images:
        row = vlm.detect_garbage(str(img))
        append_row(row)
        status = "YES" if row["garbage_detected"] else "NO "
        classes = row["classes_detected"] or "—"
        print(f"  [{status}] {img.name} | {classes} | {row['inference_s']}s | {row['vram_mb']}MB VRAM")

    vlm.unload()
    print(f"[{model_key}] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help=f"Model key or 'all'. Options: {list(REGISTRY)}")
    parser.add_argument("--images", required=True,
                        help="Image file or directory")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        print(f"No images found at: {args.images}")
        return

    keys = list(REGISTRY) if args.model == "all" else [args.model]

    for key in keys:
        if key not in REGISTRY:
            print(f"Unknown model '{key}'. Available: {list(REGISTRY)}")
            continue
        run_model(key, images)


if __name__ == "__main__":
    main()
