"""
Core pipeline: label one image → list of detections.
Batch helpers: label a folder or a full YOLO dataset split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .model import run_prompts_for_class
from .ops import mask_to_bbox, mask_to_yolo, nms
from .prompts import CLASS_DEFS

# Colours indexed by class_id (RGB)
_CLASS_COLOURS = [
    (255,  60,  60),   # 0 plastic_bottle  red
    ( 60, 200,  60),   # 1 glass           green
    ( 60, 100, 255),   # 2 can             blue
    (255, 200,   0),   # 3 plastic_bag     yellow
    (200,   0, 255),   # 4 metal_scrap     magenta
    (  0, 200, 200),   # 5 plastic_wrapper cyan
    (255, 140,   0),   # 6 trash_pile      orange
    (160, 160, 160),   # 7 trash           grey
]

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Per-image labeling
# ---------------------------------------------------------------------------

def label_image(
    image_pil: Image.Image,
    device: str,
    model_path: str = "facebook/sam3",
    det_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    min_area_ratio: float = 0.003,
    iou_threshold: float = 0.5,
    class_ids: list[int] | None = None,
    max_prompts_per_class: int = 0,
) -> list[dict]:
    """
    Run all requested classes on one image.

    Returns list of dicts:
        class_id, class_name, score, prompt, bbox (x0,y0,x1,y1), yolo (cx,cy,w,h)
    """
    if class_ids is None:
        class_ids = list(CLASS_DEFS.keys())

    img_w, img_h = image_pil.width, image_pil.height
    img_area     = img_w * img_h
    priority_map = {cid: CLASS_DEFS[cid]["priority"] for cid in CLASS_DEFS}

    all_instances: list[tuple[np.ndarray, float, str, int]] = []

    for class_id in class_ids:
        cls = CLASS_DEFS[class_id]
        instances = run_prompts_for_class(
            image_pil,
            class_id=class_id,
            prompts=cls["prompts"],
            device=device,
            model_path=model_path,
            det_threshold=det_threshold,
            mask_threshold=mask_threshold,
            max_prompts=max_prompts_per_class,
        )
        # Within-class NMS before pooling
        cls_kept = nms(instances, iou_threshold, priority_map)
        cls_kept = [i for i in cls_kept if i[0].sum() / img_area >= min_area_ratio]
        all_instances.extend(cls_kept)

    # Cross-class NMS: more-specific class wins overlapping detection
    final = nms(all_instances, iou_threshold, priority_map)

    detections = []
    for mask, score, prompt, class_id in final:
        yolo = mask_to_yolo(mask, img_w, img_h)
        bbox = mask_to_bbox(mask)
        if yolo is None:
            continue
        detections.append({
            "class_id":   class_id,
            "class_name": CLASS_DEFS[class_id]["name"],
            "score":      round(score, 4),
            "prompt":     prompt,
            "bbox":       list(bbox),
            "yolo":       [round(v, 6) for v in yolo],
        })

    return detections


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_yolo_labels(label_path: Path, detections: list[dict]) -> None:
    with open(label_path, "w") as f:
        for det in detections:
            cid = det["class_id"]
            cx, cy, w, h = det["yolo"]
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def annotate_image(image_pil: Image.Image, detections: list[dict]) -> Image.Image:
    result = image_pil.copy().convert("RGB")
    draw   = ImageDraw.Draw(result)
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        rgb   = _CLASS_COLOURS[det["class_id"] % len(_CLASS_COLOURS)]
        label = f"[{det['class_id']}] {det['class_name']} {det['score']:.2f}"
        draw.rectangle([x0, y0, x1, y1], outline=rgb, width=3)
        tw = len(label) * 7
        draw.rectangle([x0, max(0, y0 - 18), x0 + tw, max(0, y0 - 2)], fill=rgb)
        draw.text((x0 + 2, max(0, y0 - 17)), label, fill=(255, 255, 255))
    return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def label_folder(
    image_dir: Path,
    output_label_dir: Path | None,
    output_annot_dir: Path | None,
    device: str,
    model_path: str,
    det_threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
    iou_threshold: float,
    class_ids: list[int] | None,
    max_prompts_per_class: int,
    skip_existing: bool,
) -> list[dict]:
    """Label all images in a folder. Returns list of per-image report dicts."""
    try:
        from tqdm import tqdm
        wrap = tqdm
    except ImportError:
        def wrap(iterable, **_):   # fallback if tqdm not installed
            return iterable

    if output_label_dir:
        output_label_dir.mkdir(parents=True, exist_ok=True)
    if output_annot_dir:
        output_annot_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in _EXTS)
    reports: list[dict] = []

    for img_path in wrap(image_files, desc=str(image_dir), unit="img"):
        lbl_dir = output_label_dir if output_label_dir else img_path.parent
        label_path = lbl_dir / f"{img_path.stem}.txt"

        if skip_existing and label_path.exists():
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[warn] Cannot open {img_path}: {exc}")
            continue

        detections = label_image(
            image_pil,
            device=device,
            model_path=model_path,
            det_threshold=det_threshold,
            mask_threshold=mask_threshold,
            min_area_ratio=min_area_ratio,
            iou_threshold=iou_threshold,
            class_ids=class_ids,
            max_prompts_per_class=max_prompts_per_class,
        )

        write_yolo_labels(label_path, detections)

        if output_annot_dir:
            ann = annotate_image(image_pil, detections)
            ann.save(output_annot_dir / f"{img_path.stem}_ann.jpg", quality=95)

        reports.append({
            "image":      str(img_path),
            "label":      str(label_path),
            "n":          len(detections),
            "detections": detections,
        })

    return reports


def label_dataset(
    dataset_root: Path,
    output_root: Path | None,
    output_annot_root: Path | None,
    splits: list[str],
    **kwargs,
) -> list[dict]:
    """
    Process a YOLO dataset (train/val/test splits).
    Expects dataset_root/{split}/images/.
    Writes labels to dataset_root/{split}/labels/ (or output_root/{split}/labels/).
    """
    all_reports: list[dict] = []

    for split in splits:
        img_dir = dataset_root / split / "images"
        if not img_dir.exists():
            print(f"[skip] {img_dir} not found")
            continue

        lbl_dir = (
            (output_root / split / "labels") if output_root
            else (dataset_root / split / "labels")
        )
        ann_dir = (output_annot_root / split) if output_annot_root else None

        print(f"\n[{split}] {img_dir}")
        reports = label_folder(
            image_dir=img_dir,
            output_label_dir=lbl_dir,
            output_annot_dir=ann_dir,
            **kwargs,
        )
        all_reports.extend(reports)

    return all_reports
