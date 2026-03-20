"""
Water detection using Otsu's automatic thresholding.

Strategy:
  Convert to grayscale, apply Otsu's binarization to find the optimal
  threshold that separates the image into two classes (water = dark,
  sky/land = bright). Then refine with HSV hue filtering to keep
  only blue-ish regions.

  Works well when water is clearly darker than the rest of the scene
  (typical in oblique coastal views).
"""

import cv2
import numpy as np

from core.water_detector import morphological_cleanup, remove_small_regions


def create_water_mask(
    image_np: np.ndarray,
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """
    Create a water mask using Otsu's binarization on the V channel (brightness).

    Pipeline:
    1. Extract V channel from HSV.
    2. Apply Otsu's threshold → separates dark (water) from bright (sky/land).
    3. Refine: keep only pixels with blue-ish hue (removes dark land/shadows).
    4. Sky exclusion (top 10%).
    5. Morphological cleanup.

    Args:
        image_np: RGB image (H, W, 3), uint8.
        min_region_ratio: Minimum connected region area as fraction of image.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── 1. Otsu on the V (brightness) channel ────────────────────────
    # Water is typically darker than sky. Otsu finds the split automatically.
    _, otsu_mask = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # otsu_mask: 255 = dark pixels (water candidates), 0 = bright pixels

    # ── 2. Also try Otsu on the S (saturation) channel ───────────────
    # Water often has more saturation than gray sky
    _, otsu_sat = cv2.threshold(s_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: pixel is water if dark (Otsu V) OR saturated (Otsu S)
    combined = cv2.bitwise_or(otsu_mask, otsu_sat)

    # ── 3. Hue refinement: keep only blue-ish pixels ─────────────────
    # Without this, dark shadows and dark land would also pass
    hue_ok = (h_ch >= 50) & (h_ch <= 170)
    # For very desaturated pixels, accept blue-dominant in RGB
    r = image_np[:, :, 0].astype(np.float32)
    g = image_np[:, :, 1].astype(np.float32)
    b = image_np[:, :, 2].astype(np.float32)
    blue_dominant = (b >= r - 5) & (b >= g - 10) & (s_ch < 15)
    color_ok = (hue_ok | blue_dominant)

    water = combined.copy()
    water[~color_ok] = 0

    # ── 4. Sky exclusion ─────────────────────────────────────────────
    water[: int(img_h * 0.10), :] = 0
    # Exclude warm colors
    warm = ((h_ch < 30) | (h_ch > 160)) & (s_ch > 20)
    water[warm] = 0

    # ── 5. Morphological cleanup + small region removal ─────────────
    water = morphological_cleanup(water)
    water = remove_small_regions(water, min_region_ratio)

    return water
