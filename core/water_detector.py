"""
Robust water region detection for oblique coastal/harbor images.

Designed for webcam-style shots of harbors, marinas, and coastlines viewed
from elevated positions. Handles various water colors including:
- Turquoise/teal (clear harbor water)
- Deep blue (open ocean)
- Gray-blue (overcast/hazy conditions)
- Desaturated gray-blue (rough sea or flat light)

Key design decisions:
- Uses ONLY vertical edges to exclude structures (masts, buildings).
  Horizontal edges (wave crests) are intentionally preserved.
- No local variance filter: wave surface naturally has higher variance than
  what a structure-detection filter would expect.
- Adaptive saturation threshold based on global image saturation.
- Horizon detection to determine the sky exclusion zone dynamically.
"""

import cv2
import math
import random

import numpy as np


def _find_horizon_row(gray: np.ndarray) -> int:
    """
    Estimate the horizon row as the location with the strongest horizontal
    edge energy in the middle portion of the image.

    Works by finding the row with maximum summed absolute vertical gradient
    (Sobel Y), which corresponds to a strong sky-to-sea transition.

    Args:
        gray: Grayscale image (H, W), uint8.

    Returns:
        Row index of the estimated horizon.
    """
    img_h = gray.shape[0]
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5))
    row_energy = sobel_y.sum(axis=1)

    # Smooth with a box filter to reduce noise
    k = max(3, img_h // 20)
    row_energy = np.convolve(row_energy, np.ones(k) / k, mode="same")

    # Horizon should be between 15% and 70% from the top
    search_top = int(img_h * 0.15)
    search_bot = int(img_h * 0.70)

    best = search_top + int(np.argmax(row_energy[search_top:search_bot]))
    return best


def create_water_mask(
    image_np: np.ndarray,
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """
    Create a binary mask identifying water regions in a coastal image.

    Pipeline:
    1. Detect horizon to bound the sky exclusion zone.
    2. Permissive HSV color filter for water pixels.
    3. Exclude sky zone (above horizon).
    4. Exclude warm/saturated non-water pixels (land, buildings).
    5. Exclude VERTICAL edges only (structures, masts) — waves are horizontal.
    6. Morphological cleanup + small region removal.

    Args:
        image_np: RGB image as numpy array, shape (H, W, 3).
        min_region_ratio: Minimum connected region area as fraction of image.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0]  # Hue    0-180 (OpenCV)
    s_ch = hsv[:, :, 1]  # Sat    0-255
    v_ch = hsv[:, :, 2]  # Value  0-255

    # Determine if image is globally desaturated (overcast, hazy, dawn/dusk)
    global_sat = float(s_ch.mean())
    is_overcast = global_sat < 28

    # ── 1. Horizon detection ──────────────────────────────────────────────
    horizon_row = _find_horizon_row(gray)
    # Sky cutoff = horizon minus a small safety margin
    sky_cutoff = max(int(img_h * 0.12), horizon_row - 15)

    # ── 2. Color-based water candidates (permissive) ──────────────────────
    # Water hue: teal/blue range (H 70-150 in OpenCV 0-180 scale)
    # In overcast conditions lower the saturation requirement significantly
    sat_min = 3 if is_overcast else 8

    water = (
        (h_ch >= 65) & (h_ch <= 155)       # Blue-green-teal hue
        & (s_ch >= sat_min)                 # Minimal saturation
        & (v_ch >= 12) & (v_ch <= 242)      # Not pure black or blown-out
    ).astype(np.uint8) * 255

    # ── 3. Sky exclusion ─────────────────────────────────────────────────
    water[:sky_cutoff, :] = 0

    # ── 4. Warm/saturated non-water exclusion ────────────────────────────
    # Orange, red, yellow, brown pixels = land, buildings, boat hulls
    warm_sat_threshold = 15 if is_overcast else 25
    warm_excl = (
        ((h_ch < 30) | (h_ch > 155))       # Hue outside water range
        & (s_ch > warm_sat_threshold)       # Saturated enough to be non-water
    )
    water[warm_excl] = 0

    # Very bright non-blue pixels = overexposed sky, white foam (keep if strongly blue)
    bright_excl = (v_ch > 215) & ~(
        (h_ch >= 90) & (h_ch <= 140) & (s_ch > 20)
    )
    water[bright_excl] = 0

    # ── 5. Vertical edge exclusion (structures only, NOT waves) ──────────
    # Wave crests = horizontal edges (large Sobel Y, small Sobel X)
    # Masts/buildings = vertical edges (large Sobel X, small Sobel Y)
    # We only exclude vertical-dominant edges to preserve wave-covered ocean.
    gray_f = gray.astype(np.float32)
    sobel_x = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=5))
    sobel_y = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=5))

    # A pixel is a "vertical structure edge" when its X-gradient dominates
    vertical_structure = (
        (sobel_x > 25) & (sobel_x > sobel_y * 1.5)
    ).astype(np.uint8) * 255

    v_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    v_exclusion = cv2.dilate(vertical_structure, v_kernel, iterations=2)
    water[v_exclusion > 0] = 0

    # ── 6. Morphological cleanup ─────────────────────────────────────────
    # Close: fill small holes (e.g., boat wakes between water pixels)
    # Open:  remove thin noise filaments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_close)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_open)

    # ── 7. Remove small isolated regions (noise) ─────────────────────────
    min_area = int(img_h * img_w * min_region_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            water[labels == i] = 0

    return water


def find_water_positions(
    water_mask: np.ndarray,
    n_positions: int,
    object_sizes: dict,
    min_dist: int = 150,
    safety_margin: int = 40,
    max_attempts: int = 500,
) -> list:
    """
    Find random positions within water regions suitable for object placement.

    Each returned position is guaranteed to be:
    - Fully within a water region (entire object footprint checked)
    - At least `min_dist` pixels from other positions (Poisson-like spacing)
    - Inside an eroded water mask (safety_margin from water boundaries)

    Args:
        water_mask: Binary mask (H, W), 255 = water.
        n_positions: Desired number of positions.
        object_sizes: Dict {class_id: (min_w, max_w, min_h, max_h)}.
        min_dist: Minimum pixel distance between positions.
        safety_margin: Pixels to erode from water boundary.
        max_attempts: Maximum random sampling attempts.

    Returns:
        List of tuples: (cx, cy, class_id, obj_w, obj_h).
    """
    h, w = water_mask.shape

    # Erode mask so objects won't extend beyond water boundaries
    erode_size = max(5, safety_margin)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size * 2, erode_size * 2)
    )
    safe_mask = cv2.erode(water_mask, erode_kernel)

    water_pixels = np.argwhere(safe_mask > 0)  # (y, x) pairs
    if len(water_pixels) == 0:
        return []

    class_ids = list(object_sizes.keys())
    positions = []

    for _ in range(max_attempts):
        if len(positions) >= n_positions:
            break

        # Pick a random water pixel
        idx = random.randint(0, len(water_pixels) - 1)
        cy, cx = int(water_pixels[idx, 0]), int(water_pixels[idx, 1])

        # Check minimum distance from existing positions
        too_close = any(
            math.hypot(cx - px, cy - py) < min_dist
            for px, py, *_ in positions
        )
        if too_close:
            continue

        # Random class and size
        class_id = random.choice(class_ids)
        min_w, max_w, min_h, max_h = object_sizes[class_id]
        obj_w = random.randint(min_w, max_w)
        obj_h = random.randint(min_h, max_h)
        if random.random() < 0.3:
            obj_w, obj_h = obj_h, obj_w  # Occasional 90-degree rotation

        # Verify entire object footprint is within water
        oy0 = max(0, cy - obj_h // 2)
        oy1 = min(h, cy + obj_h // 2)
        ox0 = max(0, cx - obj_w // 2)
        ox1 = min(w, cx + obj_w // 2)
        if ox1 <= ox0 or oy1 <= oy0:
            continue

        region = water_mask[oy0:oy1, ox0:ox1]
        if region.mean() / 255.0 >= 0.85:  # At least 85% water coverage
            positions.append((cx, cy, class_id, obj_w, obj_h))

    return positions


def is_water_region(
    image_np: np.ndarray,
    cx: int,
    cy: int,
    half_w: int,
    half_h: int,
    margin: int = 30,
) -> bool:
    """
    Check whether a rectangular region around (cx, cy) is water.

    Legacy point-check function kept for backwards compatibility.
    For batch operations, prefer create_water_mask() + find_water_positions().
    """
    img_h, img_w = image_np.shape[:2]

    # Reject upper 20% of image (sky in oblique views)
    if cy < img_h * 0.20:
        return False

    x0 = cx - half_w - margin
    y0 = cy - half_h - margin
    x1 = cx + half_w + margin
    y1 = cy + half_h + margin

    if x0 < 0 or y0 < 0 or x1 >= img_w or y1 >= img_h:
        return False

    crop = image_np[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    avg_h = hsv[:, :, 0].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_v = hsv[:, :, 2].mean()

    gray_f = gray.astype(np.float32)
    sx = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=5))
    sy = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=5))
    # Vertical edge dominance = structure, not water
    vert_ratio = float(sx.mean()) / (float(sy.mean()) + 1e-6)

    global_sat = float(image_np[:, :, 1].mean())  # rough proxy
    sat_min = 3 if global_sat < 28 else 8

    return all([
        65 <= avg_h <= 155,     # Blue-green-teal hue
        avg_s >= sat_min,       # Minimal saturation
        12 < avg_v < 242,       # Moderate brightness
        vert_ratio < 1.5,       # Not dominated by vertical structure edges
    ])
