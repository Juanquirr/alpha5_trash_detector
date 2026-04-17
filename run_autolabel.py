"""
Auto-label trash images with SAM3. Outputs YOLO-format .txt label files.

Usage examples
--------------
# Label a folder of images (labels written alongside images)
python run_autolabel.py images/

# Specify output label dir + visual annotations for QA
python run_autolabel.py images/ --output-labels out/labels/ --output-annot out/annot/

# Process a full YOLO dataset (train/val/test splits)
python run_autolabel.py alpha5_trash_v4_dataset/ --dataset-mode

# Dataset mode with separate output dir (doesn't touch original labels)
python run_autolabel.py alpha5_trash_v4_dataset/ --dataset-mode --output-root new_dataset/

# Only label specific classes (e.g. can=2 and plastic_bottle=0)
python run_autolabel.py images/ --classes 0 2

# Use local model instead of downloading from HuggingFace
python run_autolabel.py images/ --model /path/to/sam3

# Tune thresholds
python run_autolabel.py images/ --threshold 0.25 --min-area 0.002

# Limit prompts per class (faster, less recall)
python run_autolabel.py images/ --max-prompts 3

# Re-label images that already have a .txt file
python run_autolabel.py images/ --no-skip

# List all classes and their prompts
python run_autolabel.py --list-classes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autolabel.prompts import CLASS_DEFS
from autolabel.pipeline import label_dataset, label_folder


def _print_classes() -> None:
    for cid, cls in CLASS_DEFS.items():
        print(f"\n  [{cid}] {cls['name']}  (priority={cls['priority']})")
        for p in cls["prompts"]:
            print(f"       - {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label trash images with SAM3. Outputs YOLO .txt labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Image folder (or dataset root if --dataset-mode). Omit with --list-classes.",
    )
    parser.add_argument(
        "--output-labels", "-l", default=None, metavar="DIR",
        help="Dir for output .txt labels. Default: alongside images.",
    )
    parser.add_argument(
        "--output-annot", "-a", default=None, metavar="DIR",
        help="Dir for annotated JPEG images (QA). Omit to skip.",
    )
    parser.add_argument(
        "--output-root", default=None, metavar="DIR",
        help="[dataset-mode] Root dir for output labels. Mirrors split structure.",
    )
    parser.add_argument(
        "--dataset-mode", action="store_true",
        help="Treat input as YOLO dataset root with train/val/test subdirs.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        metavar="SPLIT",
        help="[dataset-mode] Which splits to process.",
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=None, metavar="ID",
        help="Class IDs to label (0-7). Default: all.",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print class definitions and prompts then exit.",
    )
    parser.add_argument(
        "--model", default="facebook/sam3", metavar="PATH_OR_HF_ID",
        help="SAM3 model path or HuggingFace model ID.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="SAM3 instance detection confidence threshold.",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="SAM3 mask binarization threshold.",
    )
    parser.add_argument(
        "--min-area", type=float, default=0.003,
        help="Min mask area as fraction of image (0.003 = 0.3%%).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for NMS duplicate removal.",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0,
        help="Max prompts per class (0 = all). Lower = faster, less recall.",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-label images even if a .txt label already exists.",
    )
    parser.add_argument(
        "--output-json", default=None, metavar="PATH",
        help="Write full detection report to this JSON file.",
    )

    args = parser.parse_args()

    if args.list_classes:
        _print_classes()
        sys.exit(0)

    if not args.input:
        parser.error("'input' is required unless --list-classes is used.")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    common_kwargs = dict(
        device=args.device,
        model_path=args.model,
        det_threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        min_area_ratio=args.min_area,
        iou_threshold=args.iou_threshold,
        class_ids=args.classes,
        max_prompts_per_class=args.max_prompts,
        skip_existing=not args.no_skip,
    )

    print(f"Input    : {input_path.resolve()}")
    print(f"Model    : {args.model}")
    print(f"Device   : {args.device}")
    print(f"Classes  : {args.classes or 'all'}")
    print(f"threshold={args.threshold}  mask_threshold={args.mask_threshold}")
    print(f"min_area={args.min_area}  iou_threshold={args.iou_threshold}")
    print(f"max_prompts_per_class={args.max_prompts or 'all'}")
    print(f"skip_existing={not args.no_skip}\n")

    if args.dataset_mode:
        reports = label_dataset(
            dataset_root=input_path,
            output_root=Path(args.output_root) if args.output_root else None,
            output_annot_root=Path(args.output_annot) if args.output_annot else None,
            splits=args.splits,
            **common_kwargs,
        )
    else:
        reports = label_folder(
            image_dir=input_path,
            output_label_dir=Path(args.output_labels) if args.output_labels else None,
            output_annot_dir=Path(args.output_annot) if args.output_annot else None,
            **common_kwargs,
        )

    total_det = sum(r["n"] for r in reports)
    imgs_hit  = sum(1 for r in reports if r["n"] > 0)
    print(f"\nDone.  {imgs_hit}/{len(reports)} images with detections | {total_det} total labels written")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"JSON report → {out}")


if __name__ == "__main__":
    main()
