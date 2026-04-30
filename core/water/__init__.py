"""
Water detection module — 5 interchangeable strategies.

Each strategy implements:
    create_water_mask(image_np: np.ndarray) -> np.ndarray
    - Input:  RGB image as numpy array (H, W, 3)
    - Output: Binary mask (H, W), uint8 — 255 = water, 0 = not water

Available methods
-----------------
hsv     Fast HSV colour-space thresholding. Best default for clear blue/grey water.
otsu    Otsu automatic thresholding on grayscale. Good for high-contrast scenes.
kmeans  K-Means colour clustering. More robust on murky or cloudy water.
flood   Flood fill from horizon seed points. Good for open-ocean shots.
sam     SAM 3 text-prompted segmentation. Slowest but most accurate.

Typical usage
-------------
    from core.water import get_detector, find_water_positions

    detect = get_detector("hsv")
    mask = detect(np.array(image))              # numpy array 0/255
    positions = find_water_positions(mask, n=3, object_sizes=OBJECT_SIZES)
"""

import importlib

# Re-export shared utilities so callers only need to import from core.water
from core.water_detector import (        # noqa: F401
    morphological_cleanup,
    remove_small_regions,
    find_water_positions,
)

# ── Registry of available methods ────────────────────────────────────────────

AVAILABLE_METHODS = ["hsv", "otsu", "kmeans", "flood", "sam"]

_MODULES = {
    "hsv":    "core.water_detector_hsv",
    "otsu":   "core.water_detector_otsu",
    "kmeans": "core.water_detector_kmeans",
    "flood":  "core.water_detector_flood",
    "sam":    "core.water_detector_sam",
}


def get_detector(method: str):
    """Return the create_water_mask callable for the given method name.

    Args:
        method: One of 'hsv', 'otsu', 'kmeans', 'flood', 'sam'.

    Returns:
        Callable: create_water_mask(image_np: np.ndarray) -> np.ndarray

    Raises:
        ValueError: Unknown method name.
    """
    module_name = _MODULES.get(method)
    if module_name is None:
        raise ValueError(
            f"Unknown water method '{method}'. "
            f"Available: {AVAILABLE_METHODS}"
        )
    return importlib.import_module(module_name).create_water_mask
