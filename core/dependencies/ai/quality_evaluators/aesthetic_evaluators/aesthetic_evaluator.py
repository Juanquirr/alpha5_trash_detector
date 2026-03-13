from abc import abstractmethod
from PIL import Image

class AestheticQualityEvaluator:
    @abstractmethod
    def evaluate(self, image: Image.Image) -> float:
        ...


