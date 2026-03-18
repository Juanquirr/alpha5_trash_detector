"""
Robust water region detection for oblique coastal/harbor images.

Design principles (lessons learned):
- NO full Canny edge exclusion: wave crests create edges that kill the ocean.
- NO local variance filter: waves naturally have high brightness variance.
- NO unreliable horizon detection: horizon-based cutoff is too fragile.
- YES: permissive HSV + RGB blue-dominance for desaturated water.
- YES: position-aware brightness threshold to separate sky from water.
- YES: simple morphological cleanup.
"""

import cv2
import math
import random

import numpy as np


def create_water_mask(
    image_np: np.ndarray,
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """
    Create a binary mask identifying water regions in a coastal image.

    Pipeline:
    1. HSV color filter (broad blue-teal range) + RGB blue-dominance for gray water.
    2. Hard sky exclusion (top 10%).
    3. Sliding brightness threshold for the 10%-50% band
       (sky is bright + high; water is darker or lower in the image).
    4. Exclude warm-colored pixels (land, buildings).
    5. Morphological cleanup + small region removal.

    Args:
        image_np: RGB image as numpy array, shape (H, W, 3).
        min_region_ratio: Minimum connected region area as fraction of image.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0]  # 0-180
    s_ch = hsv[:, :, 1]  # 0-255
    v_ch = hsv[:, :, 2]  # 0-255

    # ── 1. Color-based water detection ─────────────────────────────────

    # A) HSV: water in the blue-teal hue range with some saturation
    hsv_water = (
        (h_ch >= 55) & (h_ch <= 165)  # Teal through blue to blue-violet
        & (s_ch >= 5)                  # Very low saturation OK
        & (v_ch >= 10) & (v_ch <= 245)
    )

    # B) RGB blue-dominance: catches desaturated gray-blue water where
    #    HSV hue becomes unreliable (S < 15 → hue is nearly arbitrary)
    r = image_np[:, :, 0].astype(np.float32)
    g = image_np[:, :, 1].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)
    blue_gray_water = (
        (s_ch < 15)                    # Very desaturated
        & (b >= r - 5)                 # Blue >= red (small tolerance)
        & (b >= g - 10)                # Blue ~>= green
        & (v_ch >= 30) & (v_ch <= 200) # Moderate brightness
    )

    water = (hsv_water | blue_gray_water).astype(np.uint8) * 255

    # ── 2. Sky rejection ──────────────────────────────────────────────

    # Hard-exclude top 10% (timestamp overlays, definite sky)
    water[: int(img_h * 0.10), :] = 0

    # Sliding brightness threshold for the 10%-50% band.
    # Rationale: sky is bright AND high in the image; water is darker.
    # - At 10% from top → exclude pixels brighter than 140
    # - At 50% from top → exclude pixels brighter than 220
    # - Below 50% → no brightness-based exclusion (not sky)
    row_idx = np.arange(img_h, dtype=np.float32).reshape(-1, 1)
    t = np.clip((row_idx / img_h - 0.10) / 0.40, 0.0, 1.0)
    brightness_ceil = 140.0 + t * 80.0  # 140 → 220

    in_sky_band = row_idx < (img_h * 0.50)
    too_bright = v_ch.astype(np.float32) > brightness_ceil
    water[in_sky_band & too_bright] = 0

    # ── 3. Warm-color exclusion (land, buildings, sunset reflections) ─
    warm_excl = ((h_ch < 30) | (h_ch > 160)) & (s_ch > 20)
    water[warm_excl] = 0

    # ── 4. Morphological cleanup ──────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_close)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_open)

    # ── 5. Remove small isolated regions (noise) ─────────────────────
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

        idx = random.randint(0, len(water_pixels) - 1)
        cy, cx = int(water_pixels[idx, 0]), int(water_pixels[idx, 1])

        too_close = any(
            math.hypot(cx - px, cy - py) < min_dist
            for px, py, *_ in positions
        )
        if too_close:
            continue

        class_id = random.choice(class_ids)
        min_w, max_w, min_h, max_h = object_sizes[class_id]
        obj_w = random.randint(min_w, max_w)
        obj_h = random.randint(min_h, max_h)
        if random.random() < 0.3:
            obj_w, obj_h = obj_h, obj_w

        oy0 = max(0, cy - obj_h // 2)
        oy1 = min(h, cy + obj_h // 2)
        ox0 = max(0, cx - obj_w // 2)
        ox1 = min(w, cx + obj_w // 2)
        if ox1 <= ox0 or oy1 <= oy0:
            continue

        region = water_mask[oy0:oy1, ox0:ox1]
        if region.mean() / 255.0 >= 0.85:
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
    Legacy point-check: is the region around (cx, cy) water?
    For batch operations, prefer create_water_mask() + find_water_positions().
    """
    img_h, img_w = image_np.shape[:2]
    if cy < img_h * 0.15:
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

    b_crop = crop[:, :, 2].astype(float)
    r_crop = crop[:, :, 0].astype(float)
    blue_dominant = b_crop.mean() >= r_crop.mean() - 5

    return (
        ((55 <= avg_h <= 165 and avg_s >= 5) or (avg_s < 15 and blue_dominant))
        and 10 < avg_v < 245
    )
