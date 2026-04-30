"""
Trash detection in images using SAM3 text prompts.

Runs SAM3's open-vocabulary segmentation to find trash, litter, and garbage
in images (e.g., ocean/harbour photos or synthetic composites).

Outputs per image:
  - annotated PNG  — coloured mask overlays + bounding boxes with scores
  - results JSON   — detections list with bbox, score, area, prompt

Usage:
    # Single image
    python sam3_trash_detection/detect.py photo.jpg

    # Batch folder
    python sam3_trash_detection/detect.py inputs/backgrounds/ --output results/

    # Also export RGBA crops of each detected instance
    python sam3_trash_detection/detect.py inputs/ --save-crops

    # Tune sensitivity
    python sam3_trash_detection/detect.py inputs/ --threshold 0.2 --min-area 0.002

    # List available prompt sets
    python sam3_trash_detection/detect.py --list-prompts
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Prompt sets — broader = more recall, narrower = higher precision
# ---------------------------------------------------------------------------
PROMPT_SETS: dict[str, list[str]] = {
    "all": [
        "trash", "garbage", "litter", "debris",
        "plastic bottle", "plastic bag", "can", "metal scrap",
        "floating trash", "ocean litter",
    ],
    "plastic": ["plastic bottle", "plastic bag", "plastic wrapper", "plastic container"],
    "general": ["trash", "garbage", "litter", "debris"],
    "ocean":   ["floating trash", "ocean litter", "floating garbage", "marine debris"],
    "cans":    ["can", "metal can", "aluminum can", "tin can"],
}

# Colour palette for up to 10 simultaneous detections (RGBA)
_COLOURS = [
    (255,  60,  60, 120),  # red
    ( 60, 200,  60, 120),  # green
    ( 60, 100, 255, 120),  # blue
    (255, 200,   0, 120),  # yellow
    (200,   0, 255, 120),  # magenta
    (  0, 200, 200, 120),  # cyan
    (255, 140,   0, 120),  # orange
    (140, 255,   0, 120),  # lime
    (255,   0, 140, 120),  # pink
    (  0, 140, 255, 120),  # sky
]

# ---------------------------------------------------------------------------
# SAM3 singleton
# ---------------------------------------------------------------------------
_model     = None
_processor = None


def _load_sam3(device: str) -> tuple:
    global _model, _processor
    if _model is None:
        from transformers import Sam3Model, Sam3Processor

        print("[SAM3] Loading model (first run may download ~3.5 GB)...")
        _processor = Sam3Processor.from_pretrained("facebook/sam3")
        _model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("[SAM3] Ready.\n")
    return _model, _processor


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_prompts(
    image_pil: Image.Image,
    prompts: list[str],
    device: str,
    det_threshold: float,
    mask_threshold: float,
) -> list[tuple[np.ndarray, float, str]]:
    """Return list of (bool_mask, score, prompt) for all prompts."""
    model, processor = _load_sam3(device)
    instances: list[tuple[np.ndarray, float, str]] = []

    for prompt in prompts:
        inputs = processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=det_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks  = results.get("masks",  [])
        scores = results.get("scores", [None] * len(masks))

        for mask_t, score_v in zip(masks, scores):
            mask_np = mask_t.cpu().numpy().astype(bool)
            score   = float(score_v) if score_v is not None else 1.0
            instances.append((mask_np, score, prompt))

    return instances


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(
    instances: list[tuple[np.ndarray, float, str]],
    iou_threshold: float,
) -> list[tuple[np.ndarray, float, str]]:
    kept: list[tuple[np.ndarray, float, str]] = []
    for mask, score, prompt in sorted(instances, key=lambda x: -x[1]):
        if all(_iou(mask, k[0]) < iou_threshold for k in kept):
            kept.append((mask, score, prompt))
    return kept


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Returns (x0, y0, x1, y1) bounding box or None if empty."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _annotate(
    image_pil: Image.Image,
    detections: list[tuple[np.ndarray, float, str]],
) -> Image.Image:
    """Draw coloured mask overlays + bounding boxes + labels."""
    overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    base    = image_pil.convert("RGBA")

    for idx, (mask, score, prompt) in enumerate(detections):
        colour = _COLOURS[idx % len(_COLOURS)]

        # Filled mask overlay
        mask_img = Image.fromarray((mask.astype(np.uint8) * colour[3]), mode="L")
        coloured  = Image.new("RGBA", image_pil.size, colour[:3] + (0,))
        coloured.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, coloured)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    draw   = ImageDraw.Draw(result)

    # Draw bboxes + labels on top
    for idx, (mask, score, prompt) in enumerate(detections):
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        rgb = _COLOURS[idx % len(_COLOURS)][:3]
        draw.rectangle([x0, y0, x1, y1], outline=rgb, width=3)

        label = f"{prompt} {score:.2f}"
        tx, ty = x0 + 4, max(0, y0 - 18)
        draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7, ty + 16], fill=rgb)
        draw.text((tx, ty), label, fill=(255, 255, 255))

    return result


def _make_rgba_crop(
    image_pil: Image.Image,
    mask: np.ndarray,
    padding: int,
    feather: int,
) -> Image.Image | None:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    h, w = image_pil.height, image_pil.width
    x0 = max(0, x0 - padding); y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding); y1 = min(h, y1 + padding)

    rgb_crop  = np.array(image_pil)[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255

    alpha = Image.fromarray(mask_crop, mode="L")
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))

    rgba       = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_crop
    rgba[:, :,  3] = np.array(alpha)
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Per-image detection
# ---------------------------------------------------------------------------

def detect_image(
    image_path: Path,
    output_dir: Path,
    prompts: list[str],
    device: str,
    det_threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
    iou_threshold: float,
    save_crops: bool,
    crop_padding: int,
    crop_feather: int,
) -> dict:
    image_pil = Image.open(image_path).convert("RGB")
    img_area  = image_pil.width * image_pil.height
    print(f"\n{'─'*60}")
    print(f"  Image  : {image_path.name}  ({image_pil.width}×{image_pil.height})")
    print(f"  Prompts: {prompts}")

    raw     = _run_prompts(image_pil, prompts, device, det_threshold, mask_threshold)
    deduped = _nms(raw, iou_threshold)
    valid   = [(m, s, p) for m, s, p in deduped if m.sum() / img_area >= min_area_ratio]

    print(f"  Detections: {len(raw)} raw → {len(deduped)} after NMS → {len(valid)} after area filter")

    report: dict = {
        "image": str(image_path),
        "size":  [image_pil.width, image_pil.height],
        "detections": [],
    }

    if not valid:
        print("  → no trash detected")
        return report

    # Annotated image
    annotated = _annotate(image_pil, valid)
    ann_path  = output_dir / f"{image_path.stem}_detected.jpg"
    annotated.save(ann_path, quality=95)
    print(f"  Annotated → {ann_path.name}")

    # RGBA crops (optional)
    crops_dir = output_dir / "crops" / image_path.stem if save_crops else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, (mask, score, prompt) in enumerate(valid):
        bbox      = _mask_bbox(mask)
        area_pct  = mask.sum() / img_area * 100
        det_entry = {
            "index":   idx,
            "prompt":  prompt,
            "score":   round(score, 4),
            "area_pct": round(area_pct, 2),
            "bbox":    list(bbox) if bbox else None,
        }

        if save_crops and crops_dir:
            rgba = _make_rgba_crop(image_pil, mask, crop_padding, crop_feather)
            if rgba is not None:
                crop_path = crops_dir / f"{idx:03d}_{prompt.replace(' ', '_')}.png"
                rgba.save(crop_path, format="PNG")
                det_entry["crop"] = str(crop_path)
                print(f"    [{idx}] score={score:.3f}  area={area_pct:.1f}%  crop→{crop_path.name}")
            else:
                print(f"    [{idx}] score={score:.3f}  area={area_pct:.1f}%  (empty crop)")
        else:
            print(f"    [{idx}] score={score:.3f}  area={area_pct:.1f}%  bbox={bbox}  prompt='{prompt}'")

        report["detections"].append(det_entry)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect trash in images using SAM3 text prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Image file or folder of images. Omit to use --list-prompts only.",
    )
    parser.add_argument(
        "--output", "-o", default="sam3_trash_detection/results",
        help="Output directory for annotated images and JSON report.",
    )
    parser.add_argument(
        "--prompt-set", choices=list(PROMPT_SETS), default="all",
        help="Which prompt set to use.",
    )
    parser.add_argument(
        "--prompts", nargs="+",
        help="Override prompt set with custom prompts (e.g. --prompts 'plastic bottle' can).",
    )
    parser.add_argument(
        "--list-prompts", action="store_true",
        help="Print all prompt sets and exit.",
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
        help="Minimum mask area as fraction of image (0.003 = 0.3%%).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for NMS duplicate removal.",
    )
    parser.add_argument(
        "--save-crops", action="store_true",
        help="Export RGBA PNG crops for each detected instance.",
    )
    parser.add_argument(
        "--crop-padding", type=int, default=10,
        help="Pixel padding around crop bounding box.",
    )
    parser.add_argument(
        "--crop-feather", type=int, default=3,
        help="Gaussian blur radius (px) for crop mask edge feathering.",
    )

    args = parser.parse_args()

    if args.list_prompts:
        print("Available prompt sets:\n")
        for name, prompts in PROMPT_SETS.items():
            print(f"  {name:10s}  {prompts}")
        sys.exit(0)

    if not args.input:
        parser.error("'input' argument required unless --list-prompts is used.")

    prompts = args.prompts if args.prompts else PROMPT_SETS[args.prompt_set]

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        image_files = [input_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)

    if not image_files:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input   : {input_path.resolve()}")
    print(f"Images  : {len(image_files)}")
    print(f"Output  : {output_dir.resolve()}")
    print(f"Prompts : {prompts}")
    print(f"Device  : {args.device}")
    print(f"threshold={args.threshold}  mask_threshold={args.mask_threshold}")
    print(f"min_area={args.min_area}  iou_threshold={args.iou_threshold}")
    if args.save_crops:
        print(f"Crops   : enabled  padding={args.crop_padding}px  feather={args.crop_feather}px")

    all_reports: list[dict] = []

    for img_path in image_files:
        report = detect_image(
            image_path=img_path,
            output_dir=output_dir,
            prompts=prompts,
            device=args.device,
            det_threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            min_area_ratio=args.min_area,
            iou_threshold=args.iou_threshold,
            save_crops=args.save_crops,
            crop_padding=args.crop_padding,
            crop_feather=args.crop_feather,
        )
        all_reports.append(report)

    # Summary JSON
    json_path = output_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n{'='*60}")
    print(f"JSON report → {json_path}")

    total_det = sum(len(r["detections"]) for r in all_reports)
    imgs_hit  = sum(1 for r in all_reports if r["detections"])
    print(f"Summary : {imgs_hit}/{len(image_files)} images with trash  |  {total_det} total detections")
    print("Done.")


if __name__ == "__main__":
    main()
