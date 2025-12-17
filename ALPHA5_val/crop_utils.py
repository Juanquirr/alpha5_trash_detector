import numpy as np
import math
import cv2
from pathlib import Path

class UniformCrops:
    """
    Generate uniform overlapping crops for an image.
    Grid adapts to image orientation (vertical images get more rows).
    """
    
    def __init__(self, overlap_ratio: float):
        """
        Args:
            overlap_ratio: Overlap between crops in [0, 1)
        """
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
        if (crops_number % 2 != 0) or (crops_number <= 0):
            raise ValueError("crops_number must be even and positive")

        coords = self._get_crops_coords(frame, crops_number)
        crops = []
        for x_min, y_min, x_max, y_max in coords:
            x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
            crops.append(frame[y1:y2, x1:x2])
        return crops, coords

    def _get_crops_coords(self, frame: np.ndarray, crops_number: int):
        """
        Calculate crop coordinates for uniform grid with overlap.
        Grid adapts to image orientation.
        
        Args:
            frame: Input image
            crops_number: Number of crops to generate
            
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        height, width = frame.shape[:2]
        is_vertical = height > width
        
        # Base grid dimensions
        base_rows = int(math.sqrt(crops_number))
        base_cols = math.ceil(crops_number / base_rows)
        
        # Swap for vertical images to get more rows than columns
        if is_vertical:
            grid_rows = base_cols
            grid_cols = base_rows
        else:
            grid_rows = base_rows
            grid_cols = base_cols

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

                x_min = c * stride_w
                y_min = r * stride_h
                x_max = x_min + cell_w
                y_max = y_min + cell_h

                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                x_max = min(float(width), x_max)
                y_max = min(float(height), y_max)

                coords.append((x_min, y_min, x_max, y_max))
                count += 1

        return coords
    

def draw_crop_grid(base_img, coords, color=(0, 255, 255), thickness=2):
    """Draw crop rectangles on image."""
    grid_img = base_img.copy()
    for x_min, y_min, x_max, y_max in coords:
        x1, y1, x2, y2 = map(int, map(round, (x_min, y_min, x_max, y_max)))
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, thickness)
    return grid_img


IMG_EXTS = {".jpg", ".jpeg", ".png"}

def iter_images(source: Path, recursive: bool):
    """List all image files from source path."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return []
