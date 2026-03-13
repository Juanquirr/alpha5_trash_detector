from PIL import Image
from abc import ABC, abstractmethod


class ImageSimilarityEvaluator(ABC):
    @abstractmethod
    def evaluate(self, image1: Image.Image, image2: Image.Image) -> float:
        ...
