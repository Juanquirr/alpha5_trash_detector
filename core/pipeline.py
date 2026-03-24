"""
Core inpainting pipeline.

Responsibilities:
- Model factory (load_model)
- Model-agnostic inpainting dispatch (run_inpaint)
- Single-object insertion, crop-based or full-image (insert_object)
- Full single-image pipeline: water detection → placement → inpainting → output (process_image)
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from core.constants import (
    DIVISOR, EDGE_MARGIN, MAX_SIDE, MIN_DIST_PX, OBJECT_SIZES,
)
from core.image_utils import (
    compute_crop_region, compute_yolo_bbox,
    create_mask, prepare_image, save_debug_image,
)
from core.water_detector import find_water_positions
import importlib

_WATER_MODULES = {
    "hsv":    "core.water_detector",
    "otsu":   "core.water_detector_otsu",
    "kmeans": "core.water_detector_kmeans",
    "flood":  "core.water_detector_flood",
    "sam":    "core.water_detector_sam",
}

def _get_water_detector(method: str):
    module_name = _WATER_MODULES.get(method)
    if module_name is None:
        raise ValueError(f"Unknown water method '{method}'. Valid: {list(_WATER_MODULES)}")
    return importlib.import_module(module_name).create_water_mask


# ── Model factory ────────────────────────────────────────────────────────────

def load_model(model_name: str, references_dir: str = "inputs/references"):
    """Instantiate an inpainting model by name (lazy import — avoids loading
    unused heavy models)."""
    if model_name == "fill":
        from core.inpainters.flux_fill import FluxLocalImageInpainter
        return FluxLocalImageInpainter()
    if model_name == "canny":
        from core.inpainters.flux_canny import FluxCannyInpainter
        return FluxCannyInpainter()
    if model_name == "redux":
        from core.inpainters.flux_redux import FluxReduxInpainter
        return FluxReduxInpainter(references_dir=references_dir)
    if model_name == "kontext":
        from core.inpainters.flux_kontext import FluxKontextInpainter
        return FluxKontextInpainter()
    raise ValueError(f"Unknown model '{model_name}'. Valid: fill, canny, redux, kontext")


# ── Low-level inpainting dispatch ────────────────────────────────────────────

def run_inpaint(
    model_name: str, model, image: Image.Image, mask: Image.Image,
    prompt: str, class_id: int,
) -> tuple[Image.Image, tuple | None]:
    """
    Call the model's inpaint() handling each model's unique interface.

    Returns:
        (result_image, bbox) — bbox is in caller's coordinate space,
        or None if the model doesn't produce its own bbox.
    """
    if model_name == "redux":
        return model.inpaint(image, mask, prompt, class_id=class_id), None
    if model_name == "kontext":
        result = model.inpaint(image, mask, prompt)
        return result, model.compute_bbox(image, result)
    return model.inpaint(image, mask, prompt), None


# ── Single-object insertion ───────────────────────────────────────────────────

def insert_object(
    model_name: str, model,
    image: Image.Image,
    cx: int, cy: int, obj_w: int, obj_h: int,
    prompt: str, class_id: int,
    use_crop: bool = True,
) -> tuple[Image.Image, tuple | None]:
    """
    Insert one object into *image* at (cx, cy).

    Two modes:
    - use_crop=True  → extract a local crop, inpaint there, paste back.
                       Better integration: model sees focused water context.
    - use_crop=False → inpaint on the full image (legacy).

    Returns:
        (updated_image, yolo_bbox) — bbox normalized to full-image coords,
        or None if it could not be computed.
    """
    img_w, img_h = image.size

    if use_crop:
        x0, y0, x1, y1 = compute_crop_region(img_w, img_h, cx, cy, obj_w, obj_h)
        crop = image.crop((x0, y0, x1, y1))
        cw, ch = crop.size

        mask = create_mask(cw, ch, cx - x0, cy - y0, obj_w, obj_h)
        result_crop, ext_bbox = run_inpaint(model_name, model, crop, mask, prompt, class_id)

        result = image.copy()
        result.paste(result_crop, (x0, y0))

        if ext_bbox:
            bxc, byc, bw, bh = ext_bbox
            bbox = ((x0 + bxc * cw) / img_w, (y0 + byc * ch) / img_h,
                    bw * cw / img_w, bh * ch / img_h)
        else:
            mask_np = np.array(mask)
            ys, xs = np.where(mask_np > 127)
            bbox = (
                (x0 + (xs.min() + xs.max()) / 2.0) / img_w,
                (y0 + (ys.min() + ys.max()) / 2.0) / img_h,
                (xs.max() - xs.min()) / img_w,
                (ys.max() - ys.min()) / img_h,
            ) if len(xs) > 0 else None

        return result, bbox

    # Full-image mode
    mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h)
    result, ext_bbox = run_inpaint(model_name, model, image, mask, prompt, class_id)
    return result, ext_bbox or compute_yolo_bbox(mask)


# ── Full single-image pipeline ────────────────────────────────────────────────

@dataclass
class ProcessConfig:
    """Runtime parameters for process_image()."""
    n_objects: int | None = None          # None → random between MIN/MAX
    use_crop: bool = True
    output_suffix: str = "_result"        # e.g. "_synth" or "_result"
    min_objects: int = 2
    max_objects: int = 3
    log_fields: list = field(default_factory=list)
    water_method: str = "hsv"             # hsv | otsu | kmeans | flood | sam
    min_water_coverage: float = 0.40      # Skip images below this water fraction (0.0-1.0)


def process_image(
    img_path: Path,
    model_name: str,
    model,
    prompts_by_class: dict,
    class_names: dict,
    out_dir: Path,
    log_writer: csv.DictWriter | None,
    cfg: ProcessConfig,
) -> list[str]:
    """
    Full pipeline for a single image:
      1. Load + resize
      2. Detect water
      3. Find object positions inside water
      4. Insert each object (inpaint)
      5. Save image, YOLO annotations, debug overlay, water mask

    Returns list of YOLO annotation strings (one per inserted object).
    """
    print(f"\n  [{img_path.name}]")

    image = Image.open(img_path).convert("RGB")
    image, scale = prepare_image(image, MAX_SIDE, DIVISOR)
    img_w, img_h = image.size
    print(f"    {img_w}x{img_h}  (scale={scale:.3f})")

    # Water detection
    create_water_mask = _get_water_detector(cfg.water_method)
    water_mask = create_water_mask(np.array(image))
    coverage = water_mask.mean() / 255.0
    print(f"    Water coverage: {coverage:.1%}")
    if coverage < cfg.min_water_coverage:
        print(f"    WARNING: water coverage {coverage:.1%} below threshold {cfg.min_water_coverage:.0%}, skipping.")
        return []

    # Object positions
    n = cfg.n_objects or random.randint(cfg.min_objects, cfg.max_objects)
    positions = find_water_positions(
        water_mask, n, OBJECT_SIZES,
        min_dist=MIN_DIST_PX, safety_margin=EDGE_MARGIN,
    )
    if not positions:
        print("    WARNING: could not place objects, skipping.")
        return []
    print(f"    Positions found: {len(positions)}")

    annotations = []
    stem = img_path.stem

    for i, (cx, cy, class_id, obj_w, obj_h) in enumerate(positions):
        prompt = random.choice(prompts_by_class[class_id])
        cls_name = class_names.get(class_id, "")
        print(f"    [{i+1}/{len(positions)}] {cls_name} at ({cx},{cy}) {obj_w}x{obj_h}px")

        image, bbox = insert_object(
            model_name, model, image,
            cx, cy, obj_w, obj_h,
            prompt, class_id, cfg.use_crop,
        )

        if bbox is None:
            continue

        xc, yc, bw, bh = bbox
        annotations.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        print(f"      bbox center=({xc:.3f},{yc:.3f}) size=({bw:.3f},{bh:.3f})")

        if log_writer:
            row = {
                "image_out": str(out_dir / f"{stem}{cfg.output_suffix}.png"),
                "source_image": img_path.name,
                "class_id": class_id,
                "class_name": cls_name,
                "prompt": prompt,
                "cx": cx, "cy": cy,
                "obj_w": obj_w, "obj_h": obj_h,
                "bbox_xc": f"{xc:.6f}", "bbox_yc": f"{yc:.6f}",
                "bbox_w": f"{bw:.6f}", "bbox_h": f"{bh:.6f}",
            }
            if "model" in cfg.log_fields:
                row["model"] = model_name
            log_writer.writerow(row)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{stem}{cfg.output_suffix}.png")
    (out_dir / f"{stem}.txt").write_text("\n".join(annotations))
    save_debug_image(image, annotations, str(out_dir / f"{stem}_debug.png"))
    Image.fromarray(water_mask).save(out_dir / f"{stem}_water_mask.png")

    print(f"    Saved → {out_dir.name}/{stem}{cfg.output_suffix}.png  ({len(annotations)} objects)")
    return annotations
