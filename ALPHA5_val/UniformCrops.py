import numpy as np
import math

class UniformCrops():
    """
    Class for uniform crops of a frame
    """
    def _init_(self, overlap_ratio: float) -> None:
        """
        Initializes crops component.
        """
        self._overlap_ratio = overlap_ratio
    
    def crop(self, frame: np.ndarray, crops_number: int) -> tuple[list[np.ndarray], list[tuple[float]]]:
        """
        Crops a frame in uniform overlapped crops
        
        :param frame: Frame to crop
        :type frame: np.ndarray
        :param crops_number: Number of crops
        :type crops_number: int
        :returns: Tuple with:
                    - List of crops
                    - List of crops coords
        :rtype: tuple[list[np.ndarray], list[tuple[float]]]
        """
        if (crops_number % 2 != 0) or (crops_number <= 0):
            raise ValueError("Crops number must be even and positive")
        
        crops_coords = self._get_crops_coords(frame, crops_number)
        crops = []
        for x_min, y_min, x_max, y_max in crops_coords:
            x_min_i = int(x_min)
            y_min_i = int(y_min)
            x_max_i = int(x_max)
            y_max_i = int(y_max)
            frame_cropped = frame[y_min_i:y_max_i, x_min_i:x_max_i]
            crops.append(frame_cropped)
        return crops, crops_coords

    def _get_crops_coords(self, frame: np.ndarray, crops_number: int) -> list[tuple[float]]:
        """
        Returns coords of the crops
        
        :param frame: Frame to get crop coords
        :type frame: np.ndarray
        :param crops_number: Number of crops
        :type crops_number: int
        :returns: A list of coords
        :rtype: list[tuple[float]]
        """
        height, width = frame.shape[:2]

        if not (0 <= self._overlap_ratio < 1):
            raise ValueError("overlap_ratio must be in [0, 1)")
        grid_rows = int(math.sqrt(crops_number))
        grid_cols = math.ceil(crops_number / grid_rows)

        cell_w = width / (grid_cols - (grid_cols - 1) * self._overlap_ratio)
        cell_h = height / (grid_rows - (grid_rows - 1) * self._overlap_ratio)

        stride_w = cell_w * (1 - self._overlap_ratio)
        stride_h = cell_h * (1 - self._overlap_ratio)

        coords: list[tuple[float]] = []
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
