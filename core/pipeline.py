"""
Core inpainting pipeline — FLUX Fill.

Current state (Phase 1):
    Only FLUX Fill is active. The inpainters/ directory also contains
    flux_redux.py and flux_canny.py, which will be wired in later phases.

Extension point:
    To add Redux or Canny, update load_model() to accept a model_name arg
    and add the corresponding branch. The rest of the pipeline is unchanged.

Data flow:
    image → water mask → placement positions → [insert_object per position]
                                                    ↓
                                            crop → object mask → FLUX Fill → paste back
                                                                    ↓
                                            YOLO annotation + debug overlay saved
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
from core.water import get_detector, find_water_positions


# ── Model factory ─────────────────────────────────────────────────────────────

def load_model():
    """Load the FLUX Fill inpainter (lazy import — GPU model only loaded here).

    Extension point: future phases will add Redux and Canny variants.
        # Phase 2: return FluxReduxInpainter(references_dir=...)
        # Phase 3: return FluxCannyInpainter()
    """
    from core.inpainters.flux_fill import FluxLocalImageInpainter
    return FluxLocalImageInpainter()


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProcessConfig:
    """Runtime parameters for process_image()."""
    n_objects: int | None = None        # None → random between min_objects/max_objects
    use_crop: bool = False              # False = full-image inpainting (background preserved)
    output_suffix: str = "_synth"
    min_objects: int = 2
    max_objects: int = 3
    log_fields: list = field(default_factory=list)
    water_method: str = "hsv"           # hsv | otsu | kmeans | flood | sam
    min_water_coverage: float = 0.40    # Skip images with less water than this
    class_filter: list | None = None    # None = all classes; [0, 3] = only those IDs
    guidance_scale: float = 30.0        # FLUX Fill classifier-free guidance (7–30 typical)
    num_inference_steps: int = 50       # Denoising steps (20 = fast, 50 = quality)


# ── Perspective-aware sizing ──────────────────────────────────────────────────

def _depth_scale(cy: int, img_h: int) -> float:
    """Scale factor based on vertical position in the frame.

    Harbour cameras show perspective: objects near the horizon (top of frame)
    are far away and appear small; objects near the bottom are close and large.

    Scale ranges from 0.6× at the horizon to 1.4× at the bottom of the frame.
    Floor prevents objects from becoming too small to generate detail.
    """
    return 0.6 + 0.8 * (cy / img_h)


# ── Single-object insertion ───────────────────────────────────────────────────

def insert_object(
    model,
    image: Image.Image,
    cx: int, cy: int, obj_w: int, obj_h: int,
    prompt: str,
    use_crop: bool = False,
    guidance_scale: float = 30.0,
    num_inference_steps: int = 50,
) -> tuple[Image.Image, tuple | None]:
    """Insert one trash object at (cx, cy) using FLUX Fill.

    Two modes:
      use_crop=False  Inpaint on the full image — background perfectly preserved.
      use_crop=True   Extract a local crop, inpaint it, paste back (faster but
                      background colour may diverge from the original scene).

    Args:
        model:               FluxLocalImageInpainter instance.
        image:               Full scene PIL image (RGB).
        cx, cy:              Centre of the placement in image pixel coordinates.
        obj_w, obj_h:        Object bounding box dimensions in pixels.
        prompt:              Text prompt describing the trash object.
        use_crop:            See above.
        guidance_scale:      CFG strength (7–30). Higher = stronger prompt adherence.
        num_inference_steps: Denoising steps (20 = fast, 50 = quality).

    Returns:
        (updated_image, yolo_bbox) — bbox normalised to [0, 1] full-image coords,
        or (image, None) if bbox could not be computed.
    """
    img_w, img_h = image.size

    if use_crop:
        x0, y0, x1, y1 = compute_crop_region(img_w, img_h, cx, cy, obj_w, obj_h)
        crop = image.crop((x0, y0, x1, y1))
        cw, ch = crop.size

        mask = create_mask(cw, ch, cx - x0, cy - y0, obj_w, obj_h)
        result_crop = model.inpaint(crop, mask, prompt,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale)
        result = image.copy()
        result.paste(result_crop, (x0, y0))

        mask_np = np.array(mask)
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return result, None
        bbox = (
            (x0 + (xs.min() + xs.max()) / 2.0) / img_w,
            (y0 + (ys.min() + ys.max()) / 2.0) / img_h,
            (xs.max() - xs.min()) / img_w,
            (ys.max() - ys.min()) / img_h,
        )
        return result, bbox

    # Full-image mode
    mask = create_mask(img_w, img_h, cx, cy, obj_w, obj_h)
    result = model.inpaint(image, mask, prompt,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale)
    return result, compute_yolo_bbox(mask)


# ── Full single-image pipeline ────────────────────────────────────────────────

def process_image(
    img_path: Path,
    model,
    prompts_by_class: dict,
    class_names: dict,
    out_dir: Path,
    log_writer: csv.DictWriter | None,
    cfg: ProcessConfig,
) -> list[str]:
    """Full pipeline for a single image.

    Steps:
      1. Load and resize the image (max side MAX_SIDE, multiple of DIVISOR)
      2. Detect water pixels with the chosen method
      3. Skip if water coverage is below cfg.min_water_coverage
      4. Find N valid object positions inside the water area
      5. Apply perspective depth scaling to object sizes
      6. Insert each object with FLUX Fill (crop mode by default)
      7. Save output image, YOLO annotations (.txt), debug overlay, water mask

    Returns:
        List of YOLO annotation strings (one per inserted object).
    """
    print(f"\n  [{img_path.name}]")

    image = Image.open(img_path).convert("RGB")
    image, scale = prepare_image(image, MAX_SIDE, DIVISOR)
    img_w, img_h = image.size
    print(f"    {img_w}x{img_h}  (scale={scale:.3f})")

    # ── 1. Water detection ────────────────────────────────────────────────────
    create_water_mask = get_detector(cfg.water_method)
    water_mask = create_water_mask(np.array(image))
    coverage = water_mask.mean() / 255.0
    print(f"    Water coverage: {coverage:.1%}")
    if coverage < cfg.min_water_coverage:
        print(f"    SKIP: {coverage:.1%} water < {cfg.min_water_coverage:.0%} threshold")
        return []

    # ── 2. Object placement ───────────────────────────────────────────────────
    # Filter to selected classes only
    active_sizes = (
        {k: v for k, v in OBJECT_SIZES.items() if k in cfg.class_filter}
        if cfg.class_filter else OBJECT_SIZES
    )
    if not active_sizes:
        print("    SKIP: no active classes after filter")
        return []

    n = cfg.n_objects or random.randint(cfg.min_objects, cfg.max_objects)
    positions = find_water_positions(
        water_mask, n, active_sizes,
        min_dist=MIN_DIST_PX, safety_margin=EDGE_MARGIN,
    )
    if not positions:
        print("    WARNING: could not find valid placement positions, skipping.")
        return []
    print(f"    Positions found: {len(positions)}")

    # ── 3. Perspective depth scaling ──────────────────────────────────────────
    # Objects lower in the frame (closer to camera) get scaled up.
    # Objects near the horizon get scaled down.
    positions = [
        (cx, cy, cls,
         max(60, int(obj_w * _depth_scale(cy, img_h))),
         max(50, int(obj_h * _depth_scale(cy, img_h))))
        for cx, cy, cls, obj_w, obj_h in positions
    ]

    # ── 4. Inpainting ─────────────────────────────────────────────────────────
    annotations = []
    stem = img_path.stem

    for i, (cx, cy, class_id, obj_w, obj_h) in enumerate(positions):
        prompt = random.choice(prompts_by_class[class_id])
        cls_name = class_names.get(class_id, "")
        depth = _depth_scale(cy, img_h)
        print(f"    [{i+1}/{len(positions)}] {cls_name} @ ({cx},{cy}) {obj_w}×{obj_h}px  depth×{depth:.2f}")

        image, bbox = insert_object(
            model, image,
            cx, cy, obj_w, obj_h,
            prompt, cfg.use_crop,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
        )

        if bbox is None:
            continue

        xc, yc, bw, bh = bbox
        annotations.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        print(f"      bbox ({xc:.3f}, {yc:.3f})  {bw:.3f}×{bh:.3f}")

        if log_writer:
            log_writer.writerow({
                "image_out":    str(out_dir / f"{stem}{cfg.output_suffix}.png"),
                "source_image": img_path.name,
                "class_id":     class_id,
                "class_name":   cls_name,
                "prompt":       prompt,
                "cx": cx, "cy": cy,
                "obj_w": obj_w, "obj_h": obj_h,
                "bbox_xc": f"{xc:.6f}", "bbox_yc": f"{yc:.6f}",
                "bbox_w":  f"{bw:.6f}", "bbox_h":  f"{bh:.6f}",
                "guidance_scale":      cfg.guidance_scale,
                "num_inference_steps": cfg.num_inference_steps,
            })

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{stem}{cfg.output_suffix}.png")
    (out_dir / f"{stem}.txt").write_text("\n".join(annotations))
    save_debug_image(image, annotations, str(out_dir / f"{stem}_debug.png"))
    Image.fromarray(water_mask).save(out_dir / f"{stem}_water_mask.png")

    print(f"    Saved → {out_dir.name}/{stem}{cfg.output_suffix}.png  ({len(annotations)} objects)")
    return annotations
