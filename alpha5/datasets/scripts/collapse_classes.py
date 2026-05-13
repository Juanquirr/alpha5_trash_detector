#!/usr/bin/env python3
"""
Collapse all class IDs in a YOLO dataset to a single target ID.

Useful for training a binary detector (e.g., "trash" vs background)
from a multi-class annotated dataset.

Input structure:
    dataset/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/   (optional)

Output structure (mirrors input):
    output/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/

Images are copied as-is. Labels have every class ID replaced with
--target_id (default: 0). Empty label files (negative samples) are
preserved unchanged.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
              ".JPG", ".JPEG", ".PNG"}
SPLITS = ("train", "val", "test")


# ─── Core logic ───────────────────────────────────────────────────────────────

def collapse_label_file(src: Path, dst: Path, target_id: int) -> tuple[int, int]:
    """
    Read a YOLO label file, replace every class ID with target_id, write output.

    Args:
        src: Source .txt label path
        dst: Destination .txt label path
        target_id: Class ID to write for every annotation

    Returns:
        (lines_kept, lines_skipped) counts
    """
    lines_kept = 0
    lines_skipped = 0
    out_lines = []

    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            lines_skipped += 1
            continue

        try:
            int(parts[0])  # validate original class ID is an integer
        except ValueError:
            lines_skipped += 1
            continue

        out_lines.append(f"{target_id} " + " ".join(parts[1:]))
        lines_kept += 1

    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""),
                   encoding="utf-8")
    return lines_kept, lines_skipped


def process_split(split_name: str, input_root: Path, output_root: Path,
                  target_id: int) -> dict:
    """
    Process one dataset split (train / val / test).

    Returns a stats dict with image/label counts.
    """
    images_src = input_root / split_name / "images"
    labels_src = input_root / split_name / "labels"

    if not images_src.exists():
        return {}

    images_dst = output_root / split_name / "images"
    labels_dst = output_root / split_name / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_copied": 0,
        "labels_converted": 0,
        "labels_empty": 0,
        "labels_missing": 0,
        "annotations_kept": 0,
        "annotations_skipped": 0,
    }

    image_files = [f for f in images_src.iterdir()
                   if f.is_file() and f.suffix in IMAGE_EXTS]

    for img_path in sorted(image_files):
        # Copy image
        shutil.copy2(img_path, images_dst / img_path.name)
        stats["images_copied"] += 1

        # Process matching label
        label_src = labels_src / (img_path.stem + ".txt")

        if not label_src.exists():
            stats["labels_missing"] += 1
            continue

        label_dst = labels_dst / (img_path.stem + ".txt")

        if label_src.stat().st_size == 0:
            # Negative sample — copy empty file as-is
            label_dst.touch()
            stats["labels_empty"] += 1
            continue

        kept, skipped = collapse_label_file(label_src, label_dst, target_id)
        stats["labels_converted"] += 1
        stats["annotations_kept"] += kept
        stats["annotations_skipped"] += skipped

    return stats


def print_stats(split: str, stats: dict) -> None:
    print(f"\n  [{split}]")
    print(f"    Images copied      : {stats['images_copied']}")
    print(f"    Labels converted   : {stats['labels_converted']}")
    print(f"    Labels empty (neg) : {stats['labels_empty']}")
    print(f"    Labels missing     : {stats['labels_missing']}")
    print(f"    Annotations kept   : {stats['annotations_kept']}")
    if stats["annotations_skipped"]:
        print(f"    Annotations skipped: {stats['annotations_skipped']}  (malformed lines)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collapse all YOLO class IDs to a single target ID. "
            "Outputs a new dataset folder with the same structure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "dataset",
        type=str,
        help="Path to YOLO dataset root (must contain train/ and/or val/ splits).",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory. Defaults to <dataset>_collapsed next to input.",
    )
    p.add_argument(
        "--target_id",
        type=int,
        default=0,
        help="Class ID to assign to every annotation (default: 0).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        metavar="SPLIT",
        help="Splits to process (default: train val test).",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()

    input_root = Path(args.dataset).resolve()
    if not input_root.exists():
        raise SystemExit(f"Dataset path not found: {input_root}")

    output_root = (
        Path(args.output).resolve()
        if args.output
        else input_root.parent / (input_root.name + "_collapsed")
    )

    if output_root == input_root:
        raise SystemExit("Output path must differ from input path.")

    print("=" * 60)
    print("COLLAPSE CLASSES")
    print("=" * 60)
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"  Target ID : {args.target_id}")
    print(f"  Splits : {args.splits}")

    found_any = False
    for split in args.splits:
        if not (input_root / split / "images").exists():
            continue
        found_any = True
        stats = process_split(split, input_root, output_root, args.target_id)
        print_stats(split, stats)

    if not found_any:
        raise SystemExit(
            f"No valid splits found in '{input_root}'. "
            f"Expected subdirectories: {args.splits}"
        )

    print(f"\nDone. Output at: {output_root}")


if __name__ == "__main__":
    main()
