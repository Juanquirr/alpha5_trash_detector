"""
Segment trash objects from reference images using SAM3.

Cuts each detected instance out of its background and saves it as a
transparent RGBA PNG, suitable for use as Redux reference images.

Usage examples:
    # Process all classes in inputs/references/ → inputs/segmented/
    python segment_references.py

    # Single class
    python segment_references.py --class can

    # Custom input directory
    python segment_references.py --input my_photos/ --class plastic_bottle

    # Lower threshold to catch harder/smaller objects
    python segment_references.py --det-threshold 0.2 --min-area 0.002

    # List supported class names
    python segment_references.py --list-classes

Output structure mirrors input:
    inputs/segmented/
        plastic_bottle/
            photo1_000.png   ← first instance found in photo1.jpg
            photo1_001.png   ← second instance (if any)
        can/
            ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Per-class text prompts fed to SAM3.
# Multiple prompts = more recall; NMS removes duplicates afterward.
# ---------------------------------------------------------------------------
_CLASS_PROMPTS: dict[str, list[str]] = {
    "plastic_bottle":  ["plastic bottle", "bottle", "plastic container"],
    "glass":           ["glass bottle", "glass jar", "bottle"],
    "can":             ["can", "metal can", "aluminum can", "tin can"],
    "plastic_bag":     ["plastic bag", "bag", "plastic film"],
    "metal_scrap":     ["metal scrap", "scrap metal", "metal debris", "rusty metal"],
    "plastic_wrapper": ["plastic wrapper", "plastic packaging", "plastic film"],
    "trash_pile":      ["trash pile", "garbage pile", "litter"],
    "trash":           ["trash", "garbage", "litter", "debris"],
}

# ---------------------------------------------------------------------------
# SAM3 singleton (loaded once, reused across all images)
# ---------------------------------------------------------------------------
_model = None
_processor = None


def _load_sam3(device: str = "cuda"):
    global _model, _processor
    if _model is None:
        from transformers import Sam3Model, Sam3Processor

        print("[SAM3] Loading model — first run may download ~3.5 GB...")
        _processor = Sam3Processor.from_pretrained("facebook/sam3")
        _model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("[SAM3] Model ready.\n")
    return _model, _processor


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def _run_sam3(
    image_pil: Image.Image,
    prompts: list[str],
    device: str,
    det_threshold: float,
    mask_threshold: float,
) -> list[tuple[np.ndarray, float, str]]:
    """
    Run SAM3 with each prompt and return raw (mask, score, prompt) triples.
    mask is a boolean ndarray of shape (H, W).
    """
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

        masks = results.get("masks", [])
        scores = results.get("scores", [None] * len(masks))

        for mask_tensor, score_val in zip(masks, scores):
            mask_np = mask_tensor.cpu().numpy().astype(bool)
            score = float(score_val) if score_val is not None else 1.0
            instances.append((mask_np, score, prompt))

    return instances


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two boolean masks."""
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(
    instances: list[tuple[np.ndarray, float, str]],
    iou_threshold: float,
) -> list[tuple[np.ndarray, float, str]]:
    """
    Greedy NMS: keep highest-scoring mask; suppress others that overlap > threshold.
    Removes duplicate detections from running multiple prompts.
    """
    kept: list[tuple[np.ndarray, float, str]] = []
    for mask, score, prompt in sorted(instances, key=lambda x: -x[1]):
        if all(_iou(mask, k[0]) < iou_threshold for k in kept):
            kept.append((mask, score, prompt))
    return kept


# ---------------------------------------------------------------------------
# RGBA crop helper
# ---------------------------------------------------------------------------

def _make_rgba_crop(
    image_pil: Image.Image,
    mask_np: np.ndarray,
    padding: int,
    feather_px: int,
) -> Image.Image | None:
    """
    Apply mask as alpha channel, blur edges for clean compositing, crop to bbox.

    Returns an RGBA PIL Image or None if the mask is empty.
    """
    h, w = image_pil.height, image_pil.width

    # Bounding box of the mask
    rows = np.where(np.any(mask_np, axis=1))[0]
    cols = np.where(np.any(mask_np, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    r0 = max(0, int(rows[0])  - padding)
    r1 = min(h, int(rows[-1]) + padding + 1)
    c0 = max(0, int(cols[0])  - padding)
    c1 = min(w, int(cols[-1]) + padding + 1)

    # Crop RGB and mask to bounding box
    rgb_crop  = np.array(image_pil)[r0:r1, c0:c1]
    mask_crop = mask_np[r0:r1, c0:c1].astype(np.uint8) * 255

    # Feather / soften edges so objects composite cleanly onto new backgrounds
    alpha_img = Image.fromarray(mask_crop, mode="L")
    if feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather_px))

    rgba = np.zeros((r1 - r0, c1 - c0, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_crop
    rgba[:, :,  3] = np.array(alpha_img)

    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Per-class processing
# ---------------------------------------------------------------------------

def _process_class(
    class_name: str,
    input_dir: Path,
    output_dir: Path,
    device: str,
    det_threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
    iou_threshold: float,
    padding: int,
    feather_px: int,
) -> None:
    prompts = _CLASS_PROMPTS.get(class_name, [class_name.replace("_", " ")])
    class_in  = input_dir / class_name
    class_out = output_dir / class_name

    if not class_in.exists():
        print(f"  [skip] {class_in} not found")
        return

    image_files = sorted(
        p for p in class_in.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not image_files:
        print(f"  [skip] no images in {class_in}")
        return

    class_out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Class : {class_name}")
    print(f"  Images: {len(image_files)}")
    print(f"  Prompts: {prompts}")
    print(f"{'='*60}")

    total_saved = 0

    for img_path in image_files:
        image_pil = Image.open(img_path).convert("RGB")
        img_area  = image_pil.width * image_pil.height
        print(f"\n  {img_path.name}  ({image_pil.width}×{image_pil.height})")

        raw = _run_sam3(image_pil, prompts, device, det_threshold, mask_threshold)

        if not raw:
            print("    → no detections")
            continue

        # Deduplicate overlapping masks from multiple prompts
        deduped = _nms(raw, iou_threshold)

        # Filter out tiny / noisy detections
        valid = [
            (m, s, p) for m, s, p in deduped
            if m.sum() / img_area >= min_area_ratio
        ]

        if not valid:
            print(f"    → {len(raw)} detection(s), all filtered (too small / duplicates)")
            continue

        print(f"    → {len(raw)} raw  |  {len(deduped)} after NMS  |  {len(valid)} after area filter")

        for idx, (mask_np, score, prompt) in enumerate(valid):
            rgba = _make_rgba_crop(image_pil, mask_np, padding, feather_px)
            if rgba is None:
                continue

            out_name = f"{img_path.stem}_{idx:03d}.png"
            rgba.save(class_out / out_name, format="PNG")
            area_pct = mask_np.sum() / img_area * 100
            print(
                f"    Saved {out_name}  "
                f"score={score:.3f}  area={area_pct:.1f}%  "
                f"size={rgba.width}×{rgba.height}  prompt='{prompt}'"
            )
            total_saved += 1

    print(f"\n  [{class_name}] Total saved: {total_saved}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment trash objects using SAM3 and save as transparent PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="inputs/references",
        help="Root directory with class subfolders containing source images.",
    )
    parser.add_argument(
        "--output", default="inputs/segmented",
        help="Root output directory. Mirrors input subfolder structure.",
    )
    parser.add_argument(
        "--class", dest="class_name", default=None,
        metavar="CLASS",
        help="Process only this class. Default: all known classes.",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print supported class names and their SAM3 prompts, then exit.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--det-threshold", type=float, default=0.3,
        help="SAM3 instance detection confidence threshold.",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="SAM3 mask binarization threshold.",
    )
    parser.add_argument(
        "--min-area", type=float, default=0.005,
        help="Minimum mask area as fraction of image (0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for NMS duplicate removal.",
    )
    parser.add_argument(
        "--padding", type=int, default=15,
        help="Pixel padding added around the bounding box crop.",
    )
    parser.add_argument(
        "--feather", type=int, default=3,
        help="Gaussian blur radius (px) for mask edge feathering.",
    )

    args = parser.parse_args()

    if args.list_classes:
        print("Supported classes and their SAM3 prompts:\n")
        for name, prompts in _CLASS_PROMPTS.items():
            print(f"  {name:20s}  {prompts}")
        sys.exit(0)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    classes = [args.class_name] if args.class_name else list(_CLASS_PROMPTS.keys())

    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {output_dir.resolve()}")
    print(f"Classes: {classes}")
    print(f"Device : {args.device}")
    print(f"det_threshold={args.det_threshold}  mask_threshold={args.mask_threshold}")
    print(f"min_area={args.min_area}  iou_threshold={args.iou_threshold}")
    print(f"padding={args.padding}px  feather={args.feather}px")

    for class_name in classes:
        _process_class(
            class_name=class_name,
            input_dir=input_dir,
            output_dir=output_dir,
            device=args.device,
            det_threshold=args.det_threshold,
            mask_threshold=args.mask_threshold,
            min_area_ratio=args.min_area,
            iou_threshold=args.iou_threshold,
            padding=args.padding,
            feather_px=args.feather,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
