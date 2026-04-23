"""
Usage:
    python run.py --model smolvlm --images images/
    python run.py --model moondream --images images/foto1.jpg
    python run.py --model all --images images/
    python run.py --model smolvlm --images images/ --limit 200   # first N images only

Venv reference:
    .transformers-5.X-venv  (transformers 5.x) : smolvlm, qwen_vl, videollama3
    .transformers-4.46-venv (transformers 4.46): moondream, llava, blip2, instructblip,
                                                 clip, paligemma, idefics, mplug_owl3

Resume: if detections_{model}.csv already exists, already-processed images are skipped.
"""
import argparse
import time
from pathlib import Path

from models import REGISTRY, VENV
from results import append_row, already_processed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def run_model(model_key: str, images: list[Path]) -> None:
    csv_path = f"results/detections_{model_key}.csv"

    # Resume: skip images already in CSV
    done = already_processed(csv_path)
    pending = [img for img in images if img.name not in done]

    if done:
        print(f"\n[{model_key}] Resume: {len(done)} already done, {len(pending)} remaining.")
    if not pending:
        print(f"[{model_key}] All images already processed. Nothing to do.")
        return

    cls = REGISTRY[model_key]
    vlm = cls()
    print(f"\n[{model_key}] Loading model...  (venv: {VENV[model_key]})")
    vlm.load()
    print(f"[{model_key}] Processing {len(pending)} image(s)...")

    t_start = time.perf_counter()
    for i, img in enumerate(pending, 1):
        row = vlm.detect_garbage(str(img))
        append_row(row, csv_path=csv_path)

        status  = "YES" if row["garbage_detected"] else "NO "
        classes = row["classes_detected"] or "—"

        # ETA
        elapsed   = time.perf_counter() - t_start
        avg_s     = elapsed / i
        remaining = avg_s * (len(pending) - i)
        eta       = _fmt_time(remaining)

        print(
            f"  [{i}/{len(pending)}] [{status}] {img.name} | {classes} "
            f"| {row['inference_s']}s | {row['vram_mb']}MB | ETA {eta}"
        )

    vlm.unload()
    total = _fmt_time(time.perf_counter() - t_start)
    print(f"[{model_key}] Done in {total}.")


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,
                        help=f"Model key or 'all'. Options: {list(REGISTRY)}")
    parser.add_argument("--images", required=True,
                        help="Image file or directory")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only first N images (useful for testing)")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        print(f"No images found at: {args.images}")
        return

    if args.limit:
        images = images[: args.limit]
        print(f"Limit: using first {len(images)} images.")

    keys = list(REGISTRY) if args.model == "all" else [args.model]

    for key in keys:
        if key not in REGISTRY:
            print(f"Unknown model '{key}'. Available: {list(REGISTRY)}")
            continue
        run_model(key, images)


if __name__ == "__main__":
    main()
