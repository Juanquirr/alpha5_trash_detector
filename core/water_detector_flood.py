"""
Water detection using flood fill segmentation.

Strategy:
  Sample seed points from the lower portion of the image (high probability
  of being water). From each seed, flood fill outward accepting pixels
  within a color tolerance. The union of all flood-filled regions = water.

  Advantages:
  - Finds connected water bodies naturally.
  - Adapts to local color variations (gradient across the water surface).
  - No global thresholds needed — tolerance is relative to the seed pixel.

  Disadvantages:
  - Depends on seed placement: bad seeds → bad mask.
  - Color tolerance (loDiff/upDiff) needs tuning per scene.
"""

import cv2
import numpy as np

from core.water_detector import morphological_cleanup, remove_small_regions


def create_water_mask(
    image_np: np.ndarray,
    n_seeds: int = 30,
    lo_diff: tuple = (15, 15, 15),
    up_diff: tuple = (15, 15, 15),
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """
    Create a water mask by flood-filling from seed points in water-likely zones.

    Pipeline:
    1. Pre-filter with bilateral filter (smooth colors, keep edges).
    2. Sample seed points from the lower 60% of the image.
    3. For each seed: check if color is plausibly water (blue-ish hue).
    4. Flood fill from valid seeds with color tolerance.
    5. Union all flood regions.
    6. Sky exclusion + hue refinement + morphological cleanup.

    Args:
        image_np: RGB image (H, W, 3), uint8.
        n_seeds: Number of seed points to try.
        lo_diff: Lower color tolerance for flood fill (B, G, R in BGR).
        up_diff: Upper color tolerance for flood fill.
        min_region_ratio: Minimum connected region area as fraction of image.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]

    # ── 1. Pre-filter to reduce noise and smooth color gradients ─────
    # Bilateral filter preserves edges while smoothing within regions
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── 2. Generate seed points in the lower 60% of the image ────────
    seed_y_min = int(img_h * 0.40)
    seed_y_max = int(img_h * 0.95)
    seed_x_min = int(img_w * 0.05)
    seed_x_max = int(img_w * 0.95)

    # Grid + jitter: evenly spaced with random offset
    grid_rows = int(np.sqrt(n_seeds))
    grid_cols = max(1, n_seeds // grid_rows)
    seeds = []
    for gy in np.linspace(seed_y_min, seed_y_max, grid_rows, dtype=int):
        for gx in np.linspace(seed_x_min, seed_x_max, grid_cols, dtype=int):
            # Small random jitter
            jy = gy + np.random.randint(-15, 16)
            jx = gx + np.random.randint(-15, 16)
            jy = np.clip(jy, seed_y_min, seed_y_max)
            jx = np.clip(jx, seed_x_min, seed_x_max)
            seeds.append((jx, jy))

    # ── 3. Filter seeds: keep only those on blue-ish pixels ──────────
    valid_seeds = []
    for sx, sy in seeds:
        sh = h_ch[sy, sx]
        ss = s_ch[sy, sx]
        sr = image_np[sy, sx, 0].astype(float)
        sb = image_np[sy, sx, 2].astype(float)

        is_blue = (50 <= sh <= 170) and ss >= 3
        is_blue_gray = ss < 15 and sb >= sr - 5

        if is_blue or is_blue_gray:
            valid_seeds.append((sx, sy))

    if not valid_seeds:
        # Fallback: try all seeds regardless of color
        valid_seeds = seeds

    # ── 4. Flood fill from each valid seed ───────────────────────────
    # floodFill modifies the mask in-place; use a shared accumulator
    flood_mask = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
    fill_image = smooth.copy()

    lo = lo_diff
    up = up_diff

    for sx, sy in valid_seeds:
        # Only fill if this pixel hasn't been filled yet
        if flood_mask[sy + 1, sx + 1] != 0:
            continue
        cv2.floodFill(
            fill_image, flood_mask, (sx, sy),
            newVal=(255, 255, 255),
            loDiff=lo, upDiff=up,
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4,
        )

    # Extract the filled region (mask has 2px border padding)
    water = flood_mask[1:-1, 1:-1]

    # ── 5. Hue refinement: only keep blue-ish flood regions ──────────
    hue_ok = (h_ch >= 45) & (h_ch <= 175)
    r = image_np[:, :, 0].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)
    blue_dom = (b >= r - 5) & (s_ch < 15)
    color_ok = hue_ok | blue_dom

    water[~color_ok] = 0

    # ── 6. Sky exclusion ─────────────────────────────────────────────
    water[: int(img_h * 0.10), :] = 0
    warm = ((h_ch < 30) | (h_ch > 160)) & (s_ch > 20)
    water[warm] = 0

    # ── 7. Morphological cleanup + small region removal ─────────────
    water = morphological_cleanup(water)
    water = remove_small_regions(water, min_region_ratio)

    return water
