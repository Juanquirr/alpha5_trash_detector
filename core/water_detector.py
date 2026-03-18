"""
Robust water region detection for oblique coastal/harbor images.

Designed for webcam-style shots of harbors, marinas, and coastlines viewed
from elevated positions. Handles various water colors (turquoise, deep blue,
gray-blue) and correctly excludes sky, buildings, boats, and land.
"""

import cv2
import math
import random

import numpy as np


def create_water_mask(
    image_np: np.ndarray,
    sky_fraction: float = 0.20,
    min_region_ratio: float = 0.005,
) -> np.ndarray:
    """
    Create a binary mask identifying water regions in a coastal image.

    Uses HSV color analysis, edge density filtering, local texture variance,
    and spatial heuristics to distinguish water from sky and structures.

    Args:
        image_np: RGB image as numpy array, shape (H, W, 3).
        sky_fraction: Fraction of image height excluded from the top (sky zone).
            In oblique coastal images, the sky occupies roughly the top 20%.
        min_region_ratio: Minimum connected component area as a fraction of total
            image area. Regions smaller than this are discarded as noise.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- 1. Per-pixel color criteria (permissive for various water types) ---
    # Hue 55-155 covers cyan, teal, blue, blue-green (OpenCV hue is 0-180)
    # Low saturation threshold allows grayish harbor water
    water = (
        (h_ch >= 55) & (h_ch <= 155)    # Blue-green-teal hue range
        & (s_ch >= 8)                    # Minimal saturation (grayish water OK)
        & (v_ch >= 20) & (v_ch <= 235)   # Not pure black or blown-out white
    ).astype(np.uint8) * 255

    # --- 2. Exclude sky zone (upper portion of oblique images) ---
    sky_cutoff = int(img_h * sky_fraction)
    water[:sky_cutoff, :] = 0

    # --- 3. Edge-based exclusion (buildings, boats, text overlays, masts) ---
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    # Dilate edges to create exclusion zones around structures
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edge_exclusion = cv2.dilate(edges, edge_kernel, iterations=2)
    water[edge_exclusion > 0] = 0

    # --- 4. Local brightness variance filter (water = smooth texture) ---
    v_float = v_ch.astype(np.float32)
    mean_v = cv2.blur(v_float, (31, 31))
    mean_v2 = cv2.blur(v_float * v_float, (31, 31))
    local_std = np.sqrt(np.maximum(mean_v2 - mean_v * mean_v, 0))
    water[local_std > 40] = 0

    # --- 5. Morphological cleanup ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_close)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_open)

    # --- 6. Remove small isolated regions (noise) ---
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

    Args:
        image_np: RGB image as numpy array.
        cx, cy: Center of the region to check.
        half_w, half_h: Half-dimensions of the region.
        margin: Extra padding around the region.

    Returns:
        True if the region appears to be water.
    """
    img_h, img_w = image_np.shape[:2]

    # Reject upper 25% of image (sky in oblique views)
    if cy < img_h * 0.25:
        return False

    x0 = cx - half_w - margin
    y0 = cy - half_h - margin
    x1 = cx + half_w + margin
    y1 = cy + half_h + margin

    if x0 < 0 or y0 < 0 or x1 >= img_w or y1 >= img_h:
        return False

    crop = image_np[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    avg_h = hsv[:, :, 0].mean()
    avg_s = hsv[:, :, 1].mean()
    avg_v = hsv[:, :, 2].mean()
    std_v = hsv[:, :, 2].std()

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = edges.mean() / 255.0

    return all([
        55 <= avg_h <= 155,     # Blue-green-teal hue
        avg_s > 8,              # Minimal saturation
        25 < avg_v < 235,       # Moderate brightness
        std_v < 55,             # Smooth texture
        edge_density < 0.12,    # Few hard edges
    ])
