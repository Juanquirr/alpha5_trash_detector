"""
Synthetic marine-trash dataset generator.

Two subcommands:

  fill   Generate a full annotated dataset using FLUX Fill (baseline model).
         Processes ALL input images. Outputs YOLO labels + debug overlays.

  test   Compare inpainting models (Canny, Redux, Kontext) on a random
         subset of images. Useful for evaluating which model integrates
         objects most naturally.

Usage:
    python run.py fill
    python run.py fill --num-instances 3 --no-crop --output outputs/

    python run.py test --model canny
    python run.py test --model all --max-images 5 --num-instances 2
    python run.py test --model redux --no-shuffle --no-crop
"""

import argparse
import csv
import random
from pathlib import Path

from core.pipeline import ProcessConfig, load_model, process_image
from core.prompts import load_class_names, load_prompts

# ── Defaults ──────────────────────────────────────────────────────────────────

INPUT_DIR    = "inputs"
PROMPTS_CSV  = "config/prompts.csv"
REFERENCES_DIR = "inputs/references"

_FILL_LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]
_TEST_LOG_FIELDS = _FILL_LOG_FIELDS + ["model"]

TEST_MODELS = ["canny", "redux", "kontext"]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_prompts():
    return load_prompts(PROMPTS_CSV), load_class_names(PROMPTS_CSV)


def _collect_images(input_dir: str, shuffle: bool, limit: int | None) -> list[Path]:
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    if shuffle:
        random.shuffle(paths)
    return paths[:limit] if limit else paths


def _open_log(path: Path, fields: list) -> tuple:
    exists = path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return fh, writer


# ── Subcommands ────────────────────────────────────────────────────────────────

def cmd_fill(args):
    """Generate a full annotated dataset with FLUX Fill."""
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_class, class_names = _load_prompts()
    image_paths = _collect_images(INPUT_DIR, shuffle=False, limit=None)

    print(f"Images: {len(image_paths)}  →  {out_dir}")
    print("Loading FLUX Fill model...")
    model = load_model("fill")

    log_fh, log_writer = _open_log(out_dir / "generation_log.csv", _FILL_LOG_FIELDS)

    cfg = ProcessConfig(
        n_objects=args.num_instances,
        use_crop=not args.no_crop,
        output_suffix="_synth",
        log_fields=_FILL_LOG_FIELDS,
        water_method=args.water_method,
        min_water_coverage=args.min_water_coverage,
    )

    for i, img_path in enumerate(image_paths):
        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(image_paths)}] {img_path.name}")
        process_image(img_path, "fill", model, prompts_by_class, class_names,
                      out_dir, log_writer, cfg)

    log_fh.close()
    print(f"\n{'─' * 60}")
    print(f"Done. Log: {out_dir}/generation_log.csv")


def cmd_test(args):
    """Compare inpainting models on a random image subset."""
    models_to_run = TEST_MODELS if args.model == "all" else [args.model]

    prompts_by_class, class_names = _load_prompts()
    image_paths = _collect_images(
        args.input, shuffle=not args.no_shuffle, limit=args.max_images
    )

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    print(f"Images selected: {[p.name for p in image_paths]}")

    out_root = Path(args.output)

    for model_name in models_to_run:
        print(f"\n{'═' * 60}")
        print(f"Model: {model_name.upper()}")
        print("Loading model...")
        model = load_model(model_name, references_dir=args.references)

        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fh, log_writer = _open_log(out_dir / "generation_log.csv", _TEST_LOG_FIELDS)

        cfg = ProcessConfig(
            n_objects=args.num_instances,
            use_crop=not args.no_crop,
            output_suffix="_result",
            log_fields=_TEST_LOG_FIELDS,
            water_method=args.water_method,
            min_water_coverage=args.min_water_coverage,
        )

        for img_path in image_paths:
            process_image(img_path, model_name, model, prompts_by_class, class_names,
                          out_dir, log_writer, cfg)

        log_fh.close()
        print(f"\n  Done → {out_dir}/")

    print(f"\n{'═' * 60}")
    print(f"All models complete. Results in {out_root}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Marine-trash synthetic dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── fill ──
    p_fill = sub.add_parser("fill", help="Generate full dataset with FLUX Fill")
    p_fill.add_argument("--output", default="outputs",
                        help="Output directory (default: outputs)")
    p_fill.add_argument("--num-instances", type=int, default=None,
                        help="Objects per image (default: random 2–3)")
    p_fill.add_argument("--no-crop", action="store_true",
                        help="Use full-image inpainting instead of crop-based")
    p_fill.add_argument("--water-method", default="hsv",
                        choices=["hsv", "otsu", "kmeans", "flood", "sam"],
                        help="Water detection method (default: hsv)")
    p_fill.add_argument("--min-water-coverage", type=float, default=0.40,
                        help="Skip images with water coverage below this fraction (default: 0.40)")

    # ── test ──
    p_test = sub.add_parser("test", help="Compare models on a subset of images")
    p_test.add_argument("--model", choices=TEST_MODELS + ["all"], default="all",
                        help="Model(s) to test (default: all)")
    p_test.add_argument("--output", default="outputs_test",
                        help="Output root directory (default: outputs_test)")
    p_test.add_argument("--input", default=INPUT_DIR,
                        help="Input image directory")
    p_test.add_argument("--max-images", type=int, default=5,
                        help="Max images per model (default: 5)")
    p_test.add_argument("--num-instances", type=int, default=1,
                        help="Objects per image (default: 1)")
    p_test.add_argument("--no-shuffle", action="store_true",
                        help="Use alphabetical order instead of random selection")
    p_test.add_argument("--no-crop", action="store_true",
                        help="Use full-image inpainting instead of crop-based")
    p_test.add_argument("--water-method", default="hsv",
                        choices=["hsv", "otsu", "kmeans", "flood", "sam"],
                        help="Water detection method (default: hsv)")
    p_test.add_argument("--min-water-coverage", type=float, default=0.40,
                        help="Skip images with water coverage below this fraction (default: 0.40)")
    p_test.add_argument("--references", default=REFERENCES_DIR,
                        help="Reference images directory for Redux (default: inputs/references)")

    args = parser.parse_args()

    if args.command == "fill":
        cmd_fill(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
