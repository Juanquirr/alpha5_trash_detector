"""
Shared water detection utilities.

This module provides:
- Post-processing functions shared by all water detector implementations.
- Position-finding for object placement within water regions.
- Re-exports create_water_mask from the default HSV detector for convenience.
"""

import cv2
import math
import random

import numpy as np


# ═══════════════════════════════════════════════════════════════
# SHARED POST-PROCESSING (used by all detector implementations)
# ═══════════════════════════════════════════════════════════════

def morphological_cleanup(
    mask: np.ndarray,
    close_size: int = 25,
    open_size: int = 11,
) -> np.ndarray:
    """Apply morphological close + open to clean up a binary mask."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def remove_small_regions(
    mask: np.ndarray,
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """Remove connected components smaller than min_region_ratio of total image area."""
    img_h, img_w = mask.shape[:2]
    min_area = int(img_h * img_w * min_region_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    return mask


# ═══════════════════════════════════════════════════════════════
# OBJECT PLACEMENT
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# DEFAULT DETECTOR RE-EXPORT
# ═══════════════════════════════════════════════════════════════

from core.water_detector_hsv import create_water_mask  # noqa: E402, F401
