"""
grounding_viz.py  —  Visualise grounding_eval.py results.

Reads grounding_{model}.csv, finds the source images, draws predicted boxes
(green) and YOLO ground-truth boxes (red) on each image, and saves annotated
copies to an output directory.

Uses the same coordinate-scale logic as grounding_eval.py:
  - max coord > 10  -> Qwen 0-1000 scale  -> divide by 1000
  - max coord > 1.0 -> pixel coordinates
  - else            -> already normalised [0, 1]

Usage:
    python vlm/grounding_viz.py --results grounding_results/grounding_qwen_vl.csv \
        --dataset alpha5/datasets/alpha7

    python vlm/grounding_viz.py --results grounding_results/grounding_qwen_2b.csv \
        --images path/to/images --labels path/to/labels

    python vlm/grounding_viz.py --results grounding_results/grounding_qwen_vl.csv \
        --dataset alpha5/datasets/alpha7 --limit 20 --out grounding_viz/
"""

import argparse
import csv
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Same patterns as parse_grounding_response in grounding_eval.py
_PATTERNS = [
    r"<ref>(.*?)</ref>\s*<box>\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)?\s*</box>",
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*<\|box_start\|>\(?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)?\s*<\|box_end\|>",
]

YOLO_CLASSES = [
    "container", "plastic", "metal", "polystyrene",
    "trash_pile", "trash",
]

# BGR colours for OpenCV
COLOR_PRED = (0, 220, 0)    # green  — predicted
COLOR_GT   = (0, 0, 220)    # red    — ground truth


def _parse_pred_boxes(pred_raw: str, img_w: int, img_h: int) -> list[dict]:
    """Parse predicted boxes from pred_raw field (same logic as grounding_eval.py)."""
    detections = []
    for pattern in _PATTERNS:
        for m in re.finditer(pattern, pred_raw, re.IGNORECASE):
            cls    = m.group(1).strip()
            coords = [float(m.group(i)) for i in range(2, 6)]

            max_coord = max(coords)
            if max_coord > 10:
                # Qwen 0-1000 scale -> normalised
                coords = [c / 1000.0 for c in coords]
            elif max_coord > 1.0:
                # Pixel coordinates -> normalised
                coords = [
                    coords[0] / img_w, coords[1] / img_h,
                    coords[2] / img_w, coords[3] / img_h,
                ]
            # else: already normalised

            coords = [max(0.0, min(1.0, c)) for c in coords]
            detections.append({"cls": cls, "box": coords})

    return detections


def _load_gt_boxes(label_path: Path) -> list[dict]:
    """Load YOLO ground-truth boxes. Returns list of {cls, box} normalised (x1,y1,x2,y2)."""
    if not label_path or not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid          = int(parts[0])
            cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        except ValueError:
            continue
        if cid >= len(YOLO_CLASSES):
            continue
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes.append({"cls": YOLO_CLASSES[cid], "box": [x1, y1, x2, y2]})
    return boxes


def _build_image_index(dataset: Path | None, images_dir: Path | None,
                       labels_dir: Path | None) -> dict[str, dict]:
    """
    Return {filename: {image: Path, label: Path | None}}.
    Searches train/val/valid/test splits when --dataset is given.
    """
    index = {}

    if dataset:
        for split in ("val", "valid", "train", "test"):
            img_dir = dataset / split / "images"
            lbl_dir = dataset / split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    lbl = lbl_dir / (img.stem + ".txt")
                    index[img.name] = {"image": img, "label": lbl if lbl.exists() else None}
    else:
        img_dir = images_dir or Path("images")
        lbl_dir = labels_dir or img_dir
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                lbl = lbl_dir / (img.stem + ".txt")
                index[img.name] = {"image": img, "label": lbl if lbl.exists() else None}

    return index


def _draw_box(img, box_norm: list[float], label: str, color: tuple, img_w: int, img_h: int):
    import cv2
    x1 = int(box_norm[0] * img_w)
    y1 = int(box_norm[1] * img_h)
    x2 = int(box_norm[2] * img_w)
    y2 = int(box_norm[3] * img_h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    ty = y1 - 4 if y1 - 4 - th >= 0 else y1 + th + 4
    cv2.rectangle(img, (x1, ty - th - baseline), (x1 + tw, ty + baseline), color, -1)
    cv2.putText(img, label, (x1, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def visualise(csv_path: Path, image_index: dict, out_dir: Path, limit: int) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing   = 0

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and processed >= limit:
                break

            img_name = row["image"]
            entry    = image_index.get(img_name)
            if not entry:
                missing += 1
                continue

            img = cv2.imread(str(entry["image"]))
            if img is None:
                missing += 1
                continue

            img_h, img_w = img.shape[:2]

            # Draw GT boxes (red)
            gt_boxes = _load_gt_boxes(entry["label"])
            for b in gt_boxes:
                _draw_box(img, b["box"], f"GT:{b['cls']}", COLOR_GT, img_w, img_h)

            # Draw predicted boxes (green)
            pred_raw = row.get("pred_raw", "")
            if pred_raw.strip().upper() != "CLEAN":
                pred_boxes = _parse_pred_boxes(pred_raw, img_w, img_h)
                for b in pred_boxes:
                    _draw_box(img, b["box"], f"P:{b['cls']}", COLOR_PRED, img_w, img_h)

            # Metrics overlay (top-left)
            prec    = row.get("precision", "?")
            rec     = row.get("recall", "?")
            miou    = row.get("mean_iou", "?")
            matched = row.get("n_matched", "?")
            n_gt    = row.get("n_gt", "?")
            n_pred  = row.get("n_pred", "?")
            info    = f"GT={n_gt} Pred={n_pred} Match={matched} | P={prec}% R={rec}% IoU={miou}"
            cv2.putText(img, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

            stem    = Path(img_name).stem
            out_path = out_dir / f"{stem}_viz.jpg"
            cv2.imwrite(str(out_path), img)
            processed += 1

    print(f"Saved {processed} images to {out_dir}")
    if missing:
        print(f"  {missing} images not found in dataset/images dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise grounding_eval.py results by drawing boxes on images."
    )
    parser.add_argument("--results", required=True,
                        help="Path to grounding_{model}.csv")
    parser.add_argument("--dataset", default=None,
                        help="YOLO dataset root (same as used with grounding_eval.py)")
    parser.add_argument("--images", default=None,
                        help="Flat images directory")
    parser.add_argument("--labels", default=None,
                        help="Flat labels directory (for GT boxes, used with --images)")
    parser.add_argument("--out", default="grounding_viz",
                        help="Output directory (default: grounding_viz)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to process (0 = all)")
    args = parser.parse_args()

    csv_path = Path(args.results)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        raise SystemExit(1)

    dataset   = Path(args.dataset)  if args.dataset else None
    images    = Path(args.images)   if args.images  else None
    labels    = Path(args.labels)   if args.labels  else None
    out_dir   = Path(args.out)

    if not dataset and not images:
        print("ERROR: provide --dataset or --images")
        raise SystemExit(1)

    print(f"Indexing images...")
    index = _build_image_index(dataset, images, labels)
    print(f"  Found {len(index)} images")

    visualise(csv_path, index, out_dir, args.limit)


if __name__ == "__main__":
    main()
