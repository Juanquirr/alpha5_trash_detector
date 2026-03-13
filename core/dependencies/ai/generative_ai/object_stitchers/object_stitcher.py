from abc import ABC, abstractmethod
from typing import Tuple, List

from PIL import Image


class ObjectStitcher(ABC):
    @abstractmethod
    def stitch(self, bg: Image.Image, fg_list: List[Image.Image], fg_mask_list: List[Image.Image],
               bbox: Tuple[int, int, int, int], num_samples: int = 1) -> Image.Image:
        ...