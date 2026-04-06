"""
Synthetic marine-trash dataset generator — FLUX Fill.

Inserts photorealistic floating trash into harbour/coastal images and
produces YOLO-format annotated training data.

Interactive usage (questionary prompts for everything):
    python run.py

Scripted usage (bypass all prompts with CLI flags):
    python run.py --input clean/ --output exp_D --classes 2 --num-instances 1 \
                  --guidance-scale 12 --steps 50 --water-method sam --max-images 3
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

INPUT_DIR   = "backgrounds"
OUTPUT_DIR  = "outputs"
PROMPTS_CSV = "config/prompts.csv"

LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
    "guidance_scale", "num_inference_steps",
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


# ── Interactive prompts (one per parameter) ───────────────────────────────────

def _ask_input_dir() -> str:
    answer = questionary.text(
        "Input folder (source images)?",
        default=INPUT_DIR,
    ).ask()
    return (answer or INPUT_DIR).strip()


def _ask_output_dir() -> str:
    answer = questionary.text(
        "Output folder (where to save results)?",
        default=OUTPUT_DIR,
    ).ask()
    return (answer or OUTPUT_DIR).strip()


def _ask_classes() -> list[int]:
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
    answer = questionary.text(
        "Objects to insert per image?",
        default="2",
        validate=lambda v: v.isdigit() and int(v) >= 1 or "Enter a positive integer",
    ).ask()
    return int(answer or "2")


def _ask_max_images() -> int | None:
    answer = questionary.text(
        "Max images to process?  (leave blank for all)",
        default="",
    ).ask()
    answer = (answer or "").strip()
    return int(answer) if answer.isdigit() else None


def _ask_water_method() -> str:
    answer = questionary.select(
        "Water detection method?",
        choices=[
            questionary.Choice("hsv    — fast, colour-based  [recommended]", value="hsv"),
            questionary.Choice("otsu   — automatic thresholding", value="otsu"),
            questionary.Choice("kmeans — colour clustering, robust", value="kmeans"),
            questionary.Choice("flood  — flood fill from horizon", value="flood"),
            questionary.Choice("sam    — SAM 3, most accurate but slow", value="sam"),
        ],
    ).ask()
    return answer or "hsv"


def _ask_min_water_coverage() -> float:
    answer = questionary.text(
        "Min water coverage to process an image?  (0.0–1.0)",
        default="0.40",
        validate=lambda v: (
            v.replace(".", "", 1).isdigit() and 0.0 <= float(v) <= 1.0
            or "Enter a number between 0.0 and 1.0"
        ),
    ).ask()
    return float(answer or "0.40")


def _ask_guidance_scale() -> float:
    answer = questionary.select(
        "Guidance scale?  (controls prompt adherence — higher = more literal)",
        choices=[
            questionary.Choice("3.5  — very loose, often blurry", value="3.5"),
            questionary.Choice("7    — soft integration", value="7"),
            questionary.Choice("10   — community sweet spot", value="10"),
            questionary.Choice("12   — good balance  [recommended]", value="12"),
            questionary.Choice("18   — strong adherence", value="18"),
            questionary.Choice("30   — maximum (default FLUX Fill)", value="30"),
        ],
    ).ask()
    return float(answer or "12")


def _ask_steps() -> int:
    answer = questionary.select(
        "Inference steps?  (more = better quality, slower)",
        choices=[
            questionary.Choice("20  — fast draft", value="20"),
            questionary.Choice("35  — balanced", value="35"),
            questionary.Choice("40  — good quality", value="40"),
            questionary.Choice("50  — full quality  [recommended]", value="50"),
        ],
    ).ask()
    return int(answer or "50")


def _ask_crop_mode() -> bool:
    answer = questionary.select(
        "Inpainting mode?",
        choices=[
            questionary.Choice(
                "Full-image  — background perfectly preserved  [recommended]",
                value="full",
            ),
            questionary.Choice(
                "Crop-based  — faster but background may diverge",
                value="crop",
            ),
        ],
    ).ask()
    return (answer or "full") == "crop"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Marine-trash synthetic dataset generator (FLUX Fill)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",          default=None)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--classes",        default=None,
                        help='"all" or comma-separated IDs, e.g. "0,2,3"')
    parser.add_argument("--num-instances",  type=int,   default=None)
    parser.add_argument("--max-images",     type=int,   default=None)
    parser.add_argument("--water-method",   default=None, choices=AVAILABLE_METHODS)
    parser.add_argument("--min-water-coverage", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--steps",          type=int,   default=None)
    parser.add_argument("--crop",           action="store_true", default=None)

    args = parser.parse_args()

    # ── Resolve: CLI flag if provided, otherwise ask interactively ────────────

    input_dir  = args.input  or _ask_input_dir()
    output_dir = args.output or _ask_output_dir()

    if args.classes is not None:
        class_filter = (
            list(ALL_CLASSES.keys()) if args.classes.lower() == "all"
            else [int(c.strip()) for c in args.classes.split(",")]
        )
    else:
        class_filter = _ask_classes()

    n_objects         = args.num_instances      if args.num_instances      is not None else _ask_num_instances()
    max_images        = args.max_images         if args.max_images         is not None else _ask_max_images()
    water_method      = args.water_method       if args.water_method       is not None else _ask_water_method()
    min_water         = args.min_water_coverage if args.min_water_coverage is not None else _ask_min_water_coverage()
    guidance_scale    = args.guidance_scale     if args.guidance_scale     is not None else _ask_guidance_scale()
    steps             = args.steps              if args.steps              is not None else _ask_steps()
    use_crop          = args.crop               if args.crop               is not None else _ask_crop_mode()

    # ── Validate ──────────────────────────────────────────────────────────────

    image_paths = _collect_images(input_dir, limit=max_images)
    if not image_paths:
        print(f"No images found in {input_dir}/")
        return

    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names      = load_class_names(PROMPTS_CSV)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_names = [ALL_CLASSES.get(c, str(c)) for c in class_filter]
    mode_label = "crop" if use_crop else "full-image"

    print()
    print(f"  Images  : {len(image_paths)}  →  {out_dir}/")
    print(f"  Classes : {', '.join(selected_names)}")
    print(f"  Objects : {n_objects} per image")
    print(f"  Water   : {water_method}  (min {min_water:.0%})")
    print(f"  FLUX    : guidance={guidance_scale}  steps={steps}  mode={mode_label}")
    print()
    print("  Loading FLUX Fill model...")

    model = load_model()

    log_fh, log_writer = _open_log(out_dir / "generation_log.csv")

    cfg = ProcessConfig(
        n_objects=n_objects,
        use_crop=use_crop,
        output_suffix="_synth",
        log_fields=LOG_FIELDS,
        water_method=water_method,
        min_water_coverage=min_water,
        class_filter=class_filter,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
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
