from typing import Tuple, List

from PIL import Image
from abc import ABC, abstractmethod

from pydantic import BaseModel


class PointExtractor(BaseModel):
    @abstractmethod
    def extract(self, image : Image.Image) -> List[Tuple[int, int]]:
        ...