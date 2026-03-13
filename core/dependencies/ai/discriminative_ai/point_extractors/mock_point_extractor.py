from typing import Tuple

from PIL import Image

from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor


class MockPointExtractor(PointExtractor):
    point : Tuple[int, int]

    def extract(self, image : Image.Image):
        return [self.point,]
