"""
Utility functions for uniform crop generation and image iteration.
"""

import math
import cv2
import numpy as np
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png"}


class UniformCrops:
    """Uniform overlapping crops for a frame."""

    def __init__(self, overlap_ratio: float) -> None:
        if not (0 <= overlap_ratio < 1):
            raise ValueError("overlap_ratio must be in [0, 1)")
        self._overlap_ratio = overlap_ratio

    def crop(self, frame: np.ndarray, crops_number: int):
        """
        Split frame into uniform overlapping crops.

        Args:
            frame: Input image (numpy array)
            crops_number: Number of crops (must be even and positive)

        Returns:
            Tuple of (crops_list, coordinates_list)
        """
        if crops_number % 2 != 0 or crops_number <= 0:
            raise ValueError("crops_number must be even and positive")
        coords = self._get_crops_coords(frame, crops_number)
        crops = []
        for x_min, y_min, x_max, y_max in coords:
            x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
            crops.append(frame[y1:y2, x1:x2])
        return crops, coords

    def _get_crops_coords(self, frame: np.ndarray, crops_number: int):
        height, width = frame.shape[:2]
        grid_rows = int(math.sqrt(crops_number))
        grid_cols = math.ceil(crops_number / grid_rows)

        cell_w = width / (grid_cols - (grid_cols - 1) * self._overlap_ratio)
        cell_h = height / (grid_rows - (grid_rows - 1) * self._overlap_ratio)
        stride_w = cell_w * (1 - self._overlap_ratio)
        stride_h = cell_h * (1 - self._overlap_ratio)

        coords = []
        count = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if count >= crops_number:
                    break
                x_min = max(0.0, c * stride_w)
                y_min = max(0.0, r * stride_h)
                x_max = min(float(width), x_min + cell_w)
                y_max = min(float(height), y_min + cell_h)
                coords.append((x_min, y_min, x_max, y_max))
                count += 1
        return coords


def draw_crop_grid(base_img: np.ndarray, coords, color=(0, 255, 255), thickness=1) -> np.ndarray:
    """
    Draw crop rectangles on a copy of the original image.

    Args:
        base_img: Original image
        coords: List of (x_min, y_min, x_max, y_max) crop coordinates
        color: BGR color for the grid lines
        thickness: Line thickness

    Returns:
        Image with crop grid drawn
    """
    grid_img = base_img.copy()
    for x_min, y_min, x_max, y_max in coords:
        x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, thickness)
    return grid_img


def iter_images(source: Path, recursive: bool = False):
    """
    List all image files from source path.

    Args:
        source: File or directory path
        recursive: Search recursively if True

    Returns:
        List of image file paths
    """
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return []
