from functools import reduce
from typing import Tuple

from PIL import Image

from dataclasses import field

from core.dependencies.utils.image.mask_generators.mask_generators import MaskGenerator


class MaskOperator(MaskGenerator):
    resolution : Tuple[int, int]
    mask_generators: list[MaskGenerator] = field(default_factory=list)

    def combine(self, mask_generator: MaskGenerator):
        if mask_generator.generate().size != self.resolution:
            raise ValueError(
                f"Resolution of mask generator {mask_generator} does not match resolution of current generator {self}"
            )
        self.mask_generators.append(mask_generator)
        return self

    def generate(self) -> Image.Image:
        base_mask = Image.new('L', self.resolution, 0)
        if len(self.mask_generators) == 0:
            return base_mask
        images = map(lambda x: x.generate(), self.mask_generators)
        final_img = reduce(lambda x, y: Image.composite(x, y, x), images)
        return final_img



