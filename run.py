"""
Synthetic marine-trash dataset generator — FLUX Fill.

Inserts photorealistic floating trash into harbour/coastal images and
produces YOLO-format annotated training data.

Interactive usage (prompts for all settings):
    python run.py

Scripted usage (all settings via CLI flags):
    python run.py --classes 0,3,7 --num-instances 2 --max-images 10
    python run.py --classes all --water-method otsu --output results/

When any of --classes / --num-instances / --water-method are omitted,
the script will ask for them interactively before starting.
"""

import argparse
import csv
import random
from pathlib import Path

import questionary

from core.pipeline import ProcessConfig, load_model, process_image
from core.prompts import load_class_names, load_prompts
from core.water import AVAILABLE_METHODS

# ── Defaults ──────────────────────────────────────────────────────────────────

INPUT_DIR   = "inputs"
OUTPUT_DIR  = "outputs"
PROMPTS_CSV = "config/prompts.csv"

LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]

ALL_CLASSES = {
    0: "plastic_bottle",
    1: "glass_bottle",
    2: "can",
    3: "plastic_bag",
    4: "metal_scrap",
    5: "plastic_wrapper",
    6: "trash_pile",
    7: "trash",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_images(input_dir: str, limit: int | None) -> list[Path]:
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    if limit:
        random.shuffle(paths)
        return paths[:limit]
    return paths


def _open_log(path: Path) -> tuple:
    exists = path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
    if not exists:
        writer.writeheader()
    return fh, writer


# ── Interactive configuration ─────────────────────────────────────────────────

def _ask_classes() -> list[int]:
    """Ask which trash classes to generate (checkbox multi-select)."""
    choices = [
        questionary.Choice(f"{name.replace('_', ' ')}  (class {cid})", value=cid)
        for cid, name in ALL_CLASSES.items()
    ]
    selected = questionary.checkbox(
        "Which trash classes to generate?",
        choices=choices,
        instruction="  space = toggle · enter = confirm",
    ).ask()
    if not selected:
        print("No classes selected — using all.")
        return list(ALL_CLASSES.keys())
    return selected


def _ask_num_instances() -> int:
    """Ask how many objects to insert per image."""
    answer = questionary.text(
        "Objects to insert per image?",
        default="2",
        validate=lambda v: v.isdigit() and int(v) >= 1 or "Enter a positive integer",
    ).ask()
    return int(answer)


def _ask_max_images() -> int | None:
    """Ask how many input images to process (blank = all, random order)."""
    answer = questionary.text(
        "How many images to process?  (leave blank for all)",
        default="",
    ).ask()
    return int(answer) if answer.strip().isdigit() else None


def _ask_water_method() -> str:
    """Ask which water detection method to use."""
    return questionary.select(
        "Water detection method?",
        choices=[
            questionary.Choice("hsv    — fast, colour-based  [recommended]", value="hsv"),
            questionary.Choice("otsu   — automatic thresholding", value="otsu"),
            questionary.Choice("kmeans — colour clustering, robust", value="kmeans"),
            questionary.Choice("flood  — flood fill from horizon", value="flood"),
            questionary.Choice("sam    — SAM 3, most accurate but slow", value="sam"),
        ],
    ).ask()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Marine-trash synthetic dataset generator (FLUX Fill)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--classes", default=None,
        help='Comma-separated class IDs (e.g. "0,3,7") or "all". '
             'Omit to select interactively.',
    )
    parser.add_argument(
        "--num-instances", type=int, default=None,
        help="Objects to insert per image. Omit to enter interactively.",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Max number of input images to process (random order). "
             "Omit for all images.",
    )
    parser.add_argument(
        "--water-method", default=None,
        choices=AVAILABLE_METHODS,
        help="Water detection method. Omit to select interactively.",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Full-image inpainting instead of crop-based (lower quality).",
    )
    parser.add_argument(
        "--min-water-coverage", type=float, default=0.40,
        help="Skip images with less water than this fraction (default: 0.40).",
    )

    args = parser.parse_args()

    # ── Resolve interactive vs CLI settings ───────────────────────────────────

    if args.classes is not None:
        if args.classes.lower() == "all":
            class_filter = list(ALL_CLASSES.keys())
        else:
            class_filter = [int(c.strip()) for c in args.classes.split(",")]
    else:
        class_filter = _ask_classes()

    n_objects = args.num_instances if args.num_instances is not None else _ask_num_instances()

    water_method = args.water_method if args.water_method is not None else _ask_water_method()

    max_images = args.max_images  # None = all (no interactive prompt for this one)

    # ── Validate and load ─────────────────────────────────────────────────────

    image_paths = _collect_images(INPUT_DIR, limit=max_images)
    if not image_paths:
        print(f"No images found in {INPUT_DIR}/")
        return

    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names      = load_class_names(PROMPTS_CSV)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_names = [ALL_CLASSES.get(c, str(c)) for c in class_filter]
    print()
    print(f"  Images  : {len(image_paths)}  →  {out_dir}/")
    print(f"  Classes : {', '.join(selected_names)}")
    print(f"  Objects : {n_objects} per image")
    print(f"  Water   : {water_method}  (min coverage {args.min_water_coverage:.0%})")
    print()
    print("  Loading FLUX Fill model...")

    model = load_model()

    log_fh, log_writer = _open_log(out_dir / "generation_log.csv")

    cfg = ProcessConfig(
        n_objects=n_objects,
        use_crop=not args.no_crop,
        output_suffix="_synth",
        log_fields=LOG_FIELDS,
        water_method=water_method,
        min_water_coverage=args.min_water_coverage,
        class_filter=class_filter,
    )

    # ── Generation loop ───────────────────────────────────────────────────────

    for i, img_path in enumerate(image_paths):
        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(image_paths)}] {img_path.name}")
        process_image(img_path, model, prompts_by_class, class_names,
                      out_dir, log_writer, cfg)

    log_fh.close()
    print(f"\n{'─' * 60}")
    print(f"Done.  {len(image_paths)} images processed.")
    print(f"Log  → {out_dir}/generation_log.csv")


if __name__ == "__main__":
    main()
