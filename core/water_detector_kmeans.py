"""
Water detection using K-Means color clustering.

Strategy:
  Cluster all pixels into K groups by color (HSV or LAB space).
  Identify which cluster(s) correspond to water based on their
  average hue (blue-teal range) and spatial position (lower half bias).

  Advantages:
  - Adapts to each image's specific color palette.
  - Handles unusual water colors that fixed thresholds might miss.
  - No manual threshold tuning for brightness/saturation.

  Disadvantages:
  - Slower than HSV thresholding (K-Means iterates).
  - K must be chosen: too low = sky+water merge, too high = fragmented.
"""

import cv2
import numpy as np


def create_water_mask(
    image_np: np.ndarray,
    n_clusters: int = 5,
    min_region_ratio: float = 0.003,
) -> np.ndarray:
    """
    Create a water mask by clustering pixels with K-Means in LAB color space.

    Pipeline:
    1. Convert to LAB, reshape to (N, 3) feature vectors.
    2. K-Means clustering into n_clusters groups.
    3. Score each cluster: hue in blue range + spatial position bias.
    4. Select water cluster(s).
    5. Sky exclusion + morphological cleanup.

    Args:
        image_np: RGB image (H, W, 3), uint8.
        n_clusters: Number of color clusters (default 5).
        min_region_ratio: Minimum connected region area as fraction of image.

    Returns:
        Binary mask (H, W), uint8: 255 = water, 0 = non-water.
    """
    img_h, img_w = image_np.shape[:2]

    # ── 1. Prepare features ──────────────────────────────────────────
    # Use LAB: perceptually uniform, better clustering than RGB/HSV
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Add y-coordinate as a feature (helps separate sky from water at same color)
    y_coords = np.arange(img_h, dtype=np.float32).reshape(-1, 1)
    y_norm = np.broadcast_to(
        y_coords / img_h * 100,  # Scale to ~0-100 to match LAB range
        (img_h, img_w),
    )

    # Feature matrix: [L, A, B, Y_position]
    features = np.stack([
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        y_norm,
    ], axis=-1).reshape(-1, 4)

    # ── 2. K-Means clustering ────────────────────────────────────────
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        features, n_clusters, None, criteria, attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    labels_2d = labels_flat.reshape(img_h, img_w)

    # ── 3. Score each cluster for "water-ness" ───────────────────────
    water = np.zeros((img_h, img_w), dtype=np.uint8)

    for cluster_id in range(n_clusters):
        cluster_mask = labels_2d == cluster_id
        if cluster_mask.sum() == 0:
            continue

        # Average HSV of this cluster
        avg_h = h_ch[cluster_mask].mean()
        avg_s = s_ch[cluster_mask].mean()
        avg_v = v_ch[cluster_mask].mean()

        # Average Y position (0 = top, 1 = bottom)
        avg_y = np.argwhere(cluster_mask)[:, 0].mean() / img_h

        # RGB blue dominance for gray clusters
        r_mean = image_np[:, :, 0][cluster_mask].astype(float).mean()
        b_mean = image_np[:, :, 2][cluster_mask].astype(float).mean()

        # Scoring: is this cluster water?
        is_blue_hue = 50 <= avg_h <= 170
        has_some_sat = avg_s >= 3
        blue_dominant = b_mean >= r_mean - 5
        is_lower_half = avg_y >= 0.30  # Water tends to be in lower portion
        not_too_bright = avg_v < 210   # Not sky-bright
        not_warm = not ((avg_h < 30 or avg_h > 160) and avg_s > 25)

        is_water = (
            (is_blue_hue and has_some_sat) or (avg_s < 15 and blue_dominant)
        ) and is_lower_half and not_too_bright and not_warm

        if is_water:
            water[cluster_mask] = 255

    # ── 4. Sky exclusion ─────────────────────────────────────────────
    water[: int(img_h * 0.10), :] = 0

    # ── 5. Morphological cleanup ─────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_close)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_open)

    # ── 6. Remove small regions ──────────────────────────────────────
    min_area = int(img_h * img_w * min_region_ratio)
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(water)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            water[labels_cc == i] = 0

    return water
