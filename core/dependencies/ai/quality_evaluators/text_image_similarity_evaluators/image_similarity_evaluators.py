from abc import abstractmethod, ABC
from PIL import Image

class TextImageSimilarityEvaluator(ABC):
    @abstractmethod
    def evaluate(self, text: str, img: Image.Image) -> float:
        ...

