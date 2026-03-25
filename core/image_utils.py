"""Shared image processing utilities for the generation pipeline."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.constants import CROP_CONTEXT_FACTOR, MIN_CROP_SIZE, MAX_CROP_SIZE


def prepare_image(image: Image.Image, max_side: int = 1024, divisor: int = 16):
    """Resize preserving aspect ratio. Round dimensions to multiples of `divisor`."""
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)  # Never upscale
    new_w = max(divisor, round(w * scale / divisor) * divisor)
    new_h = max(divisor, round(h * scale / divisor) * divisor)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def create_mask(img_w: int, img_h: int, cx: int, cy: int,
                obj_w: int, obj_h: int, blur_radius: int = 0) -> Image.Image:
    """Elliptical binary mask. Hard edges (blur_radius=0) produce sharper FLUX Fill results."""
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
