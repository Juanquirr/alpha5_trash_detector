"""
Compose segmented trash objects (RGBA PNGs) onto real water background patches
extracted from the input images. The resulting RGB images are suitable Redux
references because they show the object already floating in a marine context.

Why this matters
----------------
FluxPriorRedux encodes the *entire* appearance of a reference image. If you
pass it an isolated object on white the embeddings carry "lots of white", which
poisons FluxFill and produces colour-bleed or flat fills.  Compositing onto
real water patches gives Redux embeddings that say "object in water", which is
exactly what you want generated.

Input layout expected
---------------------
  inputs/references/{class}/*.png   ← RGBA segmented objects (from segment_references.py)
  inputs/*.jpeg (or *.jpg)          ← source water/harbour images

Output
------
  inputs/references_composed/{class}/*.jpg   ← RGB, object composited onto water

Usage
-----
    python prepare_redux_references.py                     # all classes
    python prepare_redux_references.py --class can         # single class
    python prepare_redux_references.py --n 8              # 8 compositions per object
    python prepare_redux_references.py --patch-size 512   # larger patch
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Supported classes — must match inputs/references/ subfolder names
# ---------------------------------------------------------------------------
_CLASSES = [
    "plastic_bottle",
    "glass",
    "can",
    "plastic_bag",
    "metal_scrap",
    "plastic_wrapper",
    "trash_pile",
    "trash",
]

# Object scale as fraction of patch width: (min, max)
# Larger objects get a larger slice so they remain visually prominent.
_CLASS_SCALE: dict[str, tuple[float, float]] = {
    "plastic_bottle":  (0.20, 0.40),
    "glass":           (0.20, 0.40),
    "can":             (0.18, 0.35),
    "plastic_bag":     (0.30, 0.55),
    "metal_scrap":     (0.22, 0.42),
    "plastic_wrapper": (0.25, 0.45),
    "trash_pile":      (0.40, 0.65),
    "trash":           (0.18, 0.38),
}


# ---------------------------------------------------------------------------
# Water patch extraction (CPU-only HSV — no GPU needed here)
# ---------------------------------------------------------------------------

def _hsv_water_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Fast HSV-based water mask (subset of the full HSV detector).
    Returns uint8 (H, W): 255 = water, 0 = non-water.
    """
    import cv2

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    water = (
        (h >= 55) & (h <= 165) &
        (s >= 20) &
        (v >= 20) & (v <= 230)
    )

    # Exclude top 15% (sky)
    sky_line = int(image_np.shape[0] * 0.15)
    water[:sky_line, :] = False

    mask = water.astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def _extract_water_patches(
    image_pil: Image.Image,
    patch_size: int,
    n_patches: int,
    min_water_ratio: float = 0.55,
    max_attempts: int = 60,
) -> list[Image.Image]:
    """
    Sample up to `n_patches` square crops from water regions of `image_pil`.
    Rejects patches that are less than `min_water_ratio` water.
    """
    image_np = np.array(image_pil.convert("RGB"))
    mask = _hsv_water_mask(image_np)

    h, w = image_np.shape[:2]
    if patch_size > min(h, w):
        patch_size = (min(h, w) // 16) * 16
        if patch_size < 64:
            return []

    patch_area = patch_size * patch_size
    patches: list[Image.Image] = []
    attempts = 0

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        x0 = random.randint(0, w - patch_size)
        y0 = random.randint(0, h - patch_size)
        water_ratio = mask[y0:y0 + patch_size, x0:x0 + patch_size].mean() / 255.0
        if water_ratio >= min_water_ratio:
            patches.append(image_pil.crop((x0, y0, x0 + patch_size, y0 + patch_size)))

    return patches


def _build_patch_pool(
    water_images: list[Path],
    patch_size: int,
    patches_per_image: int,
) -> list[Image.Image]:
    """Load every water image and extract patches. Returns flat list."""
    pool: list[Image.Image] = []
    for img_path in water_images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not open {img_path.name}: {e}")
            continue
        patches = _extract_water_patches(img, patch_size, patches_per_image)
        pool.extend(patches)
        print(f"  {img_path.name}: {len(patches)} patch(es)")
    return pool


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def _scale_object(
    obj_rgba: Image.Image,
    patch_size: int,
    scale_range: tuple[float, float],
) -> Image.Image:
    """
    Resize the object so its longer side is a random fraction of patch_size.
    Preserves aspect ratio.
    """
    scale = random.uniform(*scale_range)
    target_long = max(16, int(patch_size * scale))
    w, h = obj_rgba.size
    if w >= h:
        new_w = target_long
        new_h = max(1, int(h * target_long / w))
    else:
        new_h = target_long
        new_w = max(1, int(w * target_long / h))
    return obj_rgba.resize((new_w, new_h), Image.LANCZOS)


def _compose(
    patch: Image.Image,
    obj_rgba: Image.Image,
    scale_range: tuple[float, float],
    margin: int = 10,
) -> Image.Image:
    """
    Paste a scaled RGBA object onto a water patch at a random position.
    Returns an RGB image.
    """
    patch_size = patch.width  # square
    obj = _scale_object(obj_rgba, patch_size, scale_range)
    ow, oh = obj.size

    max_x = max(0, patch_size - ow - margin)
    max_y = max(0, patch_size - oh - margin)
    px = random.randint(margin, max(margin, max_x))
    py = random.randint(margin, max(margin, max_y))

    result = patch.convert("RGB").copy()
    # Use the RGBA alpha channel as paste mask
    result.paste(obj.convert("RGBA"), (px, py), mask=obj.split()[3])
    return result


# ---------------------------------------------------------------------------
# Per-class processing
# ---------------------------------------------------------------------------

def _process_class(
    class_name: str,
    ref_dir: Path,
    output_dir: Path,
    patch_pool: list[Image.Image],
    n_compositions: int,
) -> None:
    class_ref = ref_dir / class_name
    class_out = output_dir / class_name

    # Find segmented PNGs (RGBA output of segment_references.py)
    pngs = sorted(
        p for p in class_ref.iterdir()
        if p.suffix.lower() == ".png"
    ) if class_ref.exists() else []

    if not pngs:
        print(f"  [{class_name}] no segmented PNGs found — skipping")
        return

    if not patch_pool:
        print(f"  [{class_name}] patch pool is empty — skipping")
        return

    scale_range = _CLASS_SCALE.get(class_name, (0.25, 0.50))
    class_out.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{class_name}]  {len(pngs)} object(s) × {n_compositions} composition(s)")

    saved = 0
    for obj_path in pngs:
        try:
            obj_rgba = Image.open(obj_path).convert("RGBA")
        except Exception as e:
            print(f"    [warn] {obj_path.name}: {e}")
            continue

        for i in range(n_compositions):
            patch = random.choice(patch_pool)
            composed = _compose(patch, obj_rgba, scale_range)
            out_name = f"{obj_path.stem}_comp{i:02d}.jpg"
            composed.save(
                class_out / out_name,
                format="JPEG",
                quality=92,
            )
            saved += 1

    print(f"    → {saved} image(s) saved to {class_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose segmented trash onto water patches for Redux references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--references", default="inputs/references",
        help="Directory with class subfolders containing segmented PNGs.",
    )
    parser.add_argument(
        "--water-dir", default="inputs",
        help="Directory containing source water/harbour images.",
    )
    parser.add_argument(
        "--output", default="inputs/references_composed",
        help="Output directory for composed RGB references.",
    )
    parser.add_argument(
        "--class", dest="class_name", default=None, metavar="CLASS",
        help="Process only this class. Default: all.",
    )
    parser.add_argument(
        "--n", type=int, default=6,
        help="Number of compositions to generate per segmented object.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=320,
        help="Size (px) of the square water background patch.",
    )
    parser.add_argument(
        "--patches-per-image", type=int, default=4,
        help="Max water patches to extract per source image.",
    )
    parser.add_argument(
        "--min-water-ratio", type=float, default=0.55,
        help="Minimum fraction of a patch that must be water (0–1).",
    )
    args = parser.parse_args()

    ref_dir    = Path(args.references)
    water_dir  = Path(args.water_dir)
    output_dir = Path(args.output)

    if not ref_dir.exists():
        print(f"Error: references dir '{ref_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not water_dir.exists():
        print(f"Error: water dir '{water_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Collect water source images (only top-level, not subfolders)
    water_images = sorted(
        p for p in water_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not water_images:
        print(f"Error: no images found in '{water_dir}'.", file=sys.stderr)
        sys.exit(1)

    classes = [args.class_name] if args.class_name else _CLASSES

    print(f"References : {ref_dir.resolve()}")
    print(f"Water dir  : {water_dir.resolve()}  ({len(water_images)} image(s))")
    print(f"Output     : {output_dir.resolve()}")
    print(f"Classes    : {classes}")
    print(f"patch_size={args.patch_size}  patches_per_image={args.patches_per_image}  "
          f"compositions_per_object={args.n}\n")

    # Build patch pool once for all classes
    print("Extracting water patches...")
    patch_pool = _build_patch_pool(water_images, args.patch_size, args.patches_per_image)
    print(f"\nPatch pool: {len(patch_pool)} patch(es) total\n")

    if not patch_pool:
        print("Error: could not extract any water patches. "
              "Try --min-water-ratio 0.35 or check the water images.", file=sys.stderr)
        sys.exit(1)

    for class_name in classes:
        _process_class(
            class_name=class_name,
            ref_dir=ref_dir,
            output_dir=output_dir,
            patch_pool=patch_pool,
            n_compositions=args.n,
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
