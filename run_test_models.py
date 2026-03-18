"""
Comparative model testing script.

Generates test images with each model (Canny, Redux, Kontext)
to visually evaluate which one integrates trash objects best.

Usage:
    python run_test_models.py --model canny
    python run_test_models.py --model redux
    python run_test_models.py --model kontext
    python run_test_models.py --model all      # Test all three (slow)
    python run_test_models.py                   # Default: 'all'

Outputs to outputs_test/{model}/{stem}_result.png and _debug.png.
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

from core.water_detector import create_water_mask, find_water_positions

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs_test"
PROMPTS_CSV = "config/prompts.csv"
REFERENCES_DIR = "inputs/references"

MAX_SIDE = 1024
DIVISOR = 16
EDGE_MARGIN = 60
MIN_DIST_PX = 120   # Minimum spacing between objects

# Object sizes in pixels (for resized image with max side ~1024px).
# Reduced from previous iteration to improve perspective realism.
# (min_w, max_w, min_h, max_h)
OBJECT_SIZES = {
    0: (50, 100, 25, 50),     # plastic bottle - elongated
    1: (50, 100, 25, 50),     # glass bottle - similar
    2: (40, 70,  35, 65),     # can - more square
    3: (80, 150, 60, 120),    # plastic bag - larger, spread out
    4: (50, 100, 40, 80),     # metal scrap - irregular
    5: (60, 110, 40, 80),     # plastic wrapper - rectangular
    6: (120, 200, 90, 160),   # trash pile - largest
    7: (40, 80,  30, 60),     # trash - generic small
}

# Crop-based inpainting: extract a water region, inpaint at that scale,
# then paste back. This provides focused context and better integration.
CROP_CONTEXT_FACTOR = 4.0   # Crop side = N * max(obj_w, obj_h)
MIN_CROP_SIZE = 320          # Minimum crop dimension in pixels
MAX_CROP_SIZE = 640          # Maximum crop dimension in pixels

MODELS = ["canny", "redux", "kontext"]

LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt", "model",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_class_names(csv_path: str) -> dict:
    class_names = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            if cid not in class_names:
                class_names[cid] = row["class_name"].strip()
    return class_names


def load_prompts(csv_path: str) -> dict:
    prompts_by_class = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            prompt = row["prompt"].strip().strip('"')
            prompts_by_class.setdefault(cid, []).append(prompt)
    return prompts_by_class


def prepare_image(image: Image.Image, max_side: int = 1024, divisor: int = 16):
    """Resize preserving aspect ratio. Round dimensions to multiples of `divisor`."""
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(divisor, round(w * scale / divisor) * divisor)
    new_h = max(divisor, round(h * scale / divisor) * divisor)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def create_mask(img_w: int, img_h: int, cx: int, cy: int,
                obj_w: int, obj_h: int, blur_radius: int = 4) -> Image.Image:
    """Elliptical binary mask with soft edges (Gaussian blur)."""
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [cx - obj_w // 2, cy - obj_h // 2, cx + obj_w // 2, cy + obj_h // 2],
        fill=255,
    )
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def compute_yolo_bbox(mask: Image.Image) -> tuple | None:
    """Compute normalized YOLO bbox from a mask (threshold > 127)."""
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return None
    img_w, img_h = mask.size  # PIL: (w, h)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_c = ((x_min + x_max) / 2.0) / img_w
    y_c = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h


def save_debug_image(image: Image.Image, annotations: list, path):
    """Save a copy with drawn bounding boxes for visual inspection."""
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    iw, ih = debug.size
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "white"]
    for ann in annotations:
        parts = ann.split()
        cid = int(parts[0])
        xc, yc, w, h = [float(x) for x in parts[1:]]
        x0 = int((xc - w / 2) * iw)
        y0 = int((yc - h / 2) * ih)
        x1 = int((xc + w / 2) * iw)
        y1 = int((yc + h / 2) * ih)
        color = colors[cid % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"cls={cid}", fill=color)
    debug.save(path)


def compute_crop_region(img_w: int, img_h: int, cx: int, cy: int,
                        obj_w: int, obj_h: int,
                        context_factor: float = CROP_CONTEXT_FACTOR,
                        min_size: int = MIN_CROP_SIZE,
                        max_size: int = MAX_CROP_SIZE,
                        divisor: int = 16) -> tuple:
    """
    Compute a crop region centered on (cx, cy) for focused inpainting.

    The crop provides local water context around the object, allowing the
    model to generate well-integrated results at a natural scale.

    Returns:
        (crop_x0, crop_y0, crop_x1, crop_y1) in full-image pixel coordinates.
    """
    raw_size = max(obj_w, obj_h) * context_factor
    crop_side = int(max(min_size, min(max_size, raw_size)))
    crop_side = max(divisor, (crop_side // divisor) * divisor)

    # Center on target, clamp to image bounds
    crop_x0 = max(0, min(cx - crop_side // 2, img_w - crop_side))
    crop_y0 = max(0, min(cy - crop_side // 2, img_h - crop_side))
    crop_x1 = min(img_w, crop_x0 + crop_side)
    crop_y1 = min(img_h, crop_y0 + crop_side)

    # Ensure minimum size
    if crop_x1 - crop_x0 < divisor:
        crop_x1 = min(img_w, crop_x0 + divisor)
    if crop_y1 - crop_y0 < divisor:
        crop_y1 = min(img_h, crop_y0 + divisor)

    return crop_x0, crop_y0, crop_x1, crop_y1


# ═══════════════════════════════════════════════════════════════
# LAZY MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(model_name: str):
    if model_name == "canny":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_canny_inpainter import (
            FluxCannyInpainter,
        )
        return FluxCannyInpainter()

    elif model_name == "redux":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_redux_inpainter import (
            FluxReduxInpainter,
        )
        return FluxReduxInpainter(references_dir=REFERENCES_DIR)

    elif model_name == "kontext":
        from core.dependencies.ai.generative_ai.image_inpainters.flux_kontext_inpainter import (
            FluxKontextInpainter,
        )
        return FluxKontextInpainter()

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════════════════════
# INPAINTING LOGIC
# ═══════════════════════════════════════════════════════════════

def run_inpaint(model_name, model, image, mask, prompt, class_id):
    """Unified interface handling model-specific differences."""
    if model_name == "redux":
        return model.inpaint(image, mask, prompt, class_id=class_id), None
    elif model_name == "kontext":
        result = model.inpaint(image, mask, prompt)
        bbox = model.compute_bbox(image, result)
        return result, bbox
    else:
        return model.inpaint(image, mask, prompt), None


def crop_inpaint_paste(model_name, model, image, cx, cy, obj_w, obj_h,
                       prompt, class_id):
    """
    Crop-based inpainting workflow:
    1. Extract a local crop of water around the target position.
    2. Inpaint the object within the crop (model sees focused water context).
    3. Paste the result back into the full image.

    This improves integration because the model operates on a water close-up
    rather than a panoramic scene where the object would look out of scale.

    Returns:
        (result_image, yolo_bbox) where bbox is in full-image normalized coords.
    """
    img_w, img_h = image.size

    # Compute crop region
    crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_region(
        img_w, img_h, cx, cy, obj_w, obj_h
    )
    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0

    # Extract crop
    crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    # Local coordinates within the crop
    local_cx = cx - crop_x0
    local_cy = cy - crop_y0

    # Create mask in crop coordinates
    local_mask = create_mask(crop_w, crop_h, local_cx, local_cy, obj_w, obj_h)

    # Inpaint on the crop
    result_crop, external_bbox = run_inpaint(
        model_name, model, crop, local_mask, prompt, class_id
    )

    # Paste result back into full image
    result = image.copy()
    result.paste(result_crop, (crop_x0, crop_y0))

    # Compute YOLO bbox in full-image normalized coordinates
    if external_bbox:
        # Convert from crop-normalized to full-image-normalized
        bxc, byc, bw, bh = external_bbox
        full_xc = (crop_x0 + bxc * crop_w) / img_w
        full_yc = (crop_y0 + byc * crop_h) / img_h
        full_bw = (bw * crop_w) / img_w
        full_bh = (bh * crop_h) / img_h
        bbox = (full_xc, full_yc, full_bw, full_bh)
    else:
        # Compute from mask position in full-image coordinates
        mask_np = np.array(local_mask)
        ys, xs = np.where(mask_np > 127)
        if len(xs) > 0:
            xc = (crop_x0 + (xs.min() + xs.max()) / 2.0) / img_w
            yc = (crop_y0 + (ys.min() + ys.max()) / 2.0) / img_h
            bw = (xs.max() - xs.min()) / img_w
            bh = (ys.max() - ys.min()) / img_h
            bbox = (xc, yc, bw, bh)
        else:
            bbox = None

    return result, bbox


def fullimage_inpaint(model_name, model, image, cx, cy, obj_w, obj_h,
                      prompt, class_id):
    """
    Legacy full-image inpainting (no crop).
    The object is generated in the context of the entire resized image.

    Returns:
        (result_image, yolo_bbox) where bbox is in full-image normalized coords.
    """
    img_w, img_h = image.size
    mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h)

    result, external_bbox = run_inpaint(
        model_name, model, image, mask, prompt, class_id
    )

    if external_bbox:
        bbox = external_bbox
    else:
        bbox = compute_yolo_bbox(mask)

    return result, bbox


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def test_model(model_name: str, image_paths: list, prompts_by_class: dict,
               class_names: dict, num_instances: int = 1, use_crop: bool = True):
    out_dir = Path(OUTPUT_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "generation_log.csv"
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not log_exists:
        log_writer.writeheader()

    print(f"\n{'=' * 60}")
    print(f"Loading model: {model_name.upper()}")
    model = load_model(model_name)
    print(f"  OK - Model ready")

    for img_path in image_paths:
        print(f"\n  [{img_path.name}]")
        image = Image.open(img_path).convert("RGB")
        image, scale = prepare_image(image, MAX_SIDE, DIVISOR)
        img_w, img_h = image.size
        image_np = np.array(image)

        # --- Water detection ---
        print(f"    Detecting water regions...")
        water_mask = create_water_mask(image_np)
        water_coverage = water_mask.mean() / 255.0
        print(f"    Water coverage: {water_coverage:.1%}")

        if water_coverage < 0.01:
            print(f"    WARNING: No water regions found, skipping image.")
            continue

        # --- Find object positions within water ---
        positions = find_water_positions(
            water_mask, num_instances, OBJECT_SIZES,
            min_dist=MIN_DIST_PX, safety_margin=EDGE_MARGIN,
        )

        if not positions:
            print(f"    WARNING: Could not place objects in water, skipping.")
            continue

        print(f"    Found {len(positions)} valid water positions")

        stem = img_path.stem
        annotations = []

        for pos_idx, (cx, cy, class_id, obj_w, obj_h) in enumerate(positions):
            prompt = random.choice(prompts_by_class[class_id])

            print(
                f"    [{pos_idx + 1}/{len(positions)}] "
                f"class={class_id} ({class_names.get(class_id, '')}) "
                f"at ({cx},{cy}) size={obj_w}x{obj_h}px"
            )

            if use_crop:
                image, bbox = crop_inpaint_paste(
                    model_name, model, image, cx, cy, obj_w, obj_h,
                    prompt, class_id,
                )
            else:
                image, bbox = fullimage_inpaint(
                    model_name, model, image, cx, cy, obj_w, obj_h,
                    prompt, class_id,
                )

            if bbox:
                xc, yc, bw, bh = bbox
                ann = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                annotations.append(ann)
                print(f"      -> bbox: center=({xc:.3f},{yc:.3f}) size=({bw:.3f},{bh:.3f})")

            # Update numpy for next iteration
            image_np = np.array(image)

            log_writer.writerow({
                "image_out": str(out_dir / f"{stem}_result.png"),
                "source_image": img_path.name,
                "class_id": class_id,
                "class_name": class_names.get(class_id, ""),
                "prompt": prompt,
                "model": model_name,
                "cx": cx,
                "cy": cy,
                "obj_w": obj_w,
                "obj_h": obj_h,
                "bbox_xc": f"{bbox[0]:.6f}" if bbox else "",
                "bbox_yc": f"{bbox[1]:.6f}" if bbox else "",
                "bbox_w": f"{bbox[2]:.6f}" if bbox else "",
                "bbox_h": f"{bbox[3]:.6f}" if bbox else "",
            })

        image.save(out_dir / f"{stem}_result.png")
        save_debug_image(image, annotations, out_dir / f"{stem}_debug.png")
        with open(out_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(annotations))

        # Save water mask for debugging
        Image.fromarray(water_mask).save(out_dir / f"{stem}_water_mask.png")

        print(f"    OK - Saved {out_dir / stem}_result.png ({len(annotations)} objects)")

    log_file.close()
    print(f"\n  OK - Model {model_name} complete. Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparative FLUX model testing")
    parser.add_argument(
        "--model",
        choices=MODELS + ["all"],
        default="all",
        help="Model to test (canny, redux, kontext, all)",
    )
    parser.add_argument(
        "--input",
        default=INPUT_DIR,
        help="Input image directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum number of images to process per model",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of object instances to insert per image (default: 1)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable random image selection (use alphabetical order)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable crop-based inpainting (use full-image inpainting)",
    )
    args = parser.parse_args()

    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names = load_class_names(PROMPTS_CSV)

    # Collect all valid images (exclude subdirectories like references/)
    all_images = sorted(
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )

    if not all_images:
        print(f"No images found in {args.input}")
        return

    # Shuffle for random selection (unless disabled)
    if not args.no_shuffle:
        random.shuffle(all_images)

    image_paths = all_images[: args.max_images]
    print(f"Selected images: {[p.name for p in image_paths]}")

    models_to_test = MODELS if args.model == "all" else [args.model]

    for model_name in models_to_test:
        test_model(
            model_name, image_paths, prompts_by_class, class_names,
            num_instances=args.num_instances,
            use_crop=not args.no_crop,
        )

    print(f"\n{'=' * 60}")
    print(f"Tests complete. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
