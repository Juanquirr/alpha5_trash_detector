"""
Synthetic dataset generator for marine floating trash detection.

Inserts trash objects using FLUX Fill inpainting on real coastal images,
generating YOLO-format annotations for training object detectors.

Usage:
    python run_fill.py
    python run_fill.py --num-instances 3
    python run_fill.py --no-crop   # Legacy full-image inpainting
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
from core.dependencies.ai.generative_ai.image_inpainters.flux_local_image_inpainter import (
    FluxLocalImageInpainter,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
PROMPTS_CSV = "config/prompts.csv"

MAX_SIDE = 1024       # Max side after resize (preserves aspect ratio)
DIVISOR = 16          # FLUX requires dimensions as multiples of 16

MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_DIST_PX = 120     # Minimum spacing between objects
EDGE_MARGIN = 60      # Safety margin from water boundaries

# Object sizes in pixels (for resized image with max side ~1024px).
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

# Crop-based inpainting settings
CROP_CONTEXT_FACTOR = 4.0
MIN_CROP_SIZE = 320
MAX_CROP_SIZE = 640

LOG_FIELDS = [
    "image_out", "source_image",
    "class_id", "class_name", "prompt",
    "cx", "cy", "obj_w", "obj_h",
    "bbox_xc", "bbox_yc", "bbox_w", "bbox_h",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_prompts(csv_path: str) -> dict:
    prompts_by_class = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            prompt = row["prompt"].strip().strip('"')
            prompts_by_class.setdefault(cid, []).append(prompt)
    return prompts_by_class


def load_class_names(csv_path: str) -> dict:
    class_names = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            if cid not in class_names:
                class_names[cid] = row["class_name"].strip()
    return class_names


def prepare_image(image: Image.Image, max_side: int = 1024, divisor: int = 16):
    """Resize preserving aspect ratio. Round dimensions to multiples of `divisor`."""
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)  # Never upscale
    new_w = max(divisor, round(w * scale / divisor) * divisor)
    new_h = max(divisor, round(h * scale / divisor) * divisor)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def create_mask(img_w: int, img_h: int, cx: int, cy: int,
                obj_w: int, obj_h: int, blur_radius: int = 4) -> Image.Image:
    """Elliptical binary mask with soft edges (Gaussian blur)."""
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [cx - obj_w // 2, cy - obj_h // 2,
         cx + obj_w // 2, cy + obj_h // 2],
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


def save_debug_image(image: Image.Image, annotations: list, path: str):
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

    Returns:
        (crop_x0, crop_y0, crop_x1, crop_y1) in full-image pixel coordinates.
    """
    raw_size = max(obj_w, obj_h) * context_factor
    crop_side = int(max(min_size, min(max_size, raw_size)))
    crop_side = max(divisor, (crop_side // divisor) * divisor)

    crop_x0 = max(0, min(cx - crop_side // 2, img_w - crop_side))
    crop_y0 = max(0, min(cy - crop_side // 2, img_h - crop_side))
    crop_x1 = min(img_w, crop_x0 + crop_side)
    crop_y1 = min(img_h, crop_y0 + crop_side)

    if crop_x1 - crop_x0 < divisor:
        crop_x1 = min(img_w, crop_x0 + divisor)
    if crop_y1 - crop_y0 < divisor:
        crop_y1 = min(img_h, crop_y0 + divisor)

    return crop_x0, crop_y0, crop_x1, crop_y1


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic trash dataset generator with FLUX Fill"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Exact number of instances per image (default: random 2-3)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable crop-based inpainting (use full-image inpainting)",
    )
    args = parser.parse_args()

    use_crop = not args.no_crop

    # 1. Load prompts and class names
    print("Loading prompts from CSV...")
    prompts_by_class = load_prompts(PROMPTS_CSV)
    class_names = load_class_names(PROMPTS_CSV)
    class_ids = list(prompts_by_class.keys())
    print(f"  Available classes: {len(class_ids)}")

    # 2. Load model
    print("Loading FLUX Fill (may take ~30s)...")
    inpainter = FluxLocalImageInpainter()
    print("  OK - Model loaded")

    # 3. Open log CSV (append; header only if new)
    LOG_CSV = f"{OUTPUT_DIR}/generation_log.csv"
    log_exists = Path(LOG_CSV).exists()
    log_file = open(LOG_CSV, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if not log_exists:
        log_writer.writeheader()

    # 4. List input images (exclude subdirectories like references/)
    image_paths = sorted(
        p for p in Path(INPUT_DIR).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    print(f"Input images: {len(image_paths)}")

    # 5. Process each image
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n{'=' * 60}")
        print(f"[{img_idx + 1}/{len(image_paths)}] {img_path.name}")

        image = Image.open(img_path).convert("RGB")
        print(f"  Original: {image.size[0]}x{image.size[1]}")

        image, scale = prepare_image(image, max_side=MAX_SIDE, divisor=DIVISOR)
        img_w, img_h = image.size
        print(f"  Resized: {img_w}x{img_h} (scale={scale:.3f})")

        image_np = np.array(image)

        # Water detection
        print(f"  Detecting water regions...")
        water_mask = create_water_mask(image_np)
        water_coverage = water_mask.mean() / 255.0
        print(f"  Water coverage: {water_coverage:.1%}")

        if water_coverage < 0.01:
            print(f"  WARNING: No water regions found, skipping image.")
            continue

        # Find positions within water
        n_objects = (
            args.num_instances
            if args.num_instances is not None
            else random.randint(MIN_OBJECTS, MAX_OBJECTS)
        )
        positions = find_water_positions(
            water_mask, n_objects, OBJECT_SIZES,
            min_dist=MIN_DIST_PX, safety_margin=EDGE_MARGIN,
        )
        print(f"  Valid water positions: {len(positions)}")

        if not positions:
            print(f"  WARNING: Could not place objects in water, skipping image.")
            continue

        # Insert objects
        annotations = []
        for pos_idx, (cx, cy, class_id, obj_w, obj_h) in enumerate(positions):
            prompt = random.choice(prompts_by_class[class_id])

            print(
                f"  [{pos_idx + 1}/{len(positions)}] "
                f"class={class_id} ({class_names.get(class_id, '')}) "
                f"at ({cx},{cy}), size={obj_w}x{obj_h}px"
            )

            if use_crop:
                # Crop-based inpainting for better integration
                crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_region(
                    img_w, img_h, cx, cy, obj_w, obj_h
                )
                crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                crop_w, crop_h = crop.size
                local_cx = cx - crop_x0
                local_cy = cy - crop_y0
                local_mask = create_mask(
                    crop_w, crop_h, local_cx, local_cy, obj_w, obj_h
                )

                result_crop = inpainter.inpaint(crop, local_mask, prompt)
                image.paste(result_crop, (crop_x0, crop_y0))

                # YOLO bbox in full-image normalized coordinates
                xc = cx / img_w
                yc = cy / img_h
                bw = obj_w / img_w
                bh = obj_h / img_h
            else:
                # Legacy full-image inpainting
                mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h, blur_radius=4)
                image = inpainter.inpaint(image, mask, prompt)

                bbox = compute_yolo_bbox(mask)
                if bbox:
                    xc, yc, bw, bh = bbox
                else:
                    continue

            image_np = np.array(image)

            annotations.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            print(f"    -> bbox: center=({xc:.3f},{yc:.3f}) size=({bw:.3f},{bh:.3f})")

            log_writer.writerow({
                "image_out": f"{OUTPUT_DIR}/{img_path.stem}_synth.png",
                "source_image": img_path.name,
                "class_id": class_id,
                "class_name": class_names.get(class_id, ""),
                "prompt": prompt,
                "cx": cx,
                "cy": cy,
                "obj_w": obj_w,
                "obj_h": obj_h,
                "bbox_xc": f"{xc:.6f}",
                "bbox_yc": f"{yc:.6f}",
                "bbox_w": f"{bw:.6f}",
                "bbox_h": f"{bh:.6f}",
            })

        # Save results
        stem = img_path.stem
        image.save(f"{OUTPUT_DIR}/{stem}_synth.png")
        with open(f"{OUTPUT_DIR}/{stem}_synth.txt", "w") as f:
            f.write("\n".join(annotations))
        save_debug_image(image, annotations, f"{OUTPUT_DIR}/{stem}_debug.png")

        # Save water mask for debugging
        Image.fromarray(water_mask).save(f"{OUTPUT_DIR}/{stem}_water_mask.png")

        print(f"  OK - {OUTPUT_DIR}/{stem}_synth.png ({len(annotations)} objects)")

    log_file.close()
    print(f"\n{'=' * 60}")
    print(f"Generation complete. Log: {LOG_CSV}")


if __name__ == "__main__":
    main()
