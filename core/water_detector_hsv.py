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
import numpy as np

from core.water_detector import morphological_cleanup, remove_small_regions


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
    row_idx = np.arange(img_h, dtype=np.float32).reshape(-1, 1)
    t = np.clip((row_idx / img_h - 0.10) / 0.40, 0.0, 1.0)
    brightness_ceil = 140.0 + t * 80.0  # 140 → 220

    in_sky_band = row_idx < (img_h * 0.50)
    too_bright = v_ch.astype(np.float32) > brightness_ceil
    water[in_sky_band & too_bright] = 0

    # ── 3. Warm-color exclusion (land, buildings, sunset reflections) ─
    warm_excl = ((h_ch < 30) | (h_ch > 160)) & (s_ch > 20)
    water[warm_excl] = 0

    # ── 4. Morphological cleanup + small region removal ───────────────
    water = morphological_cleanup(water)
    water = remove_small_regions(water, min_region_ratio)

    return water
