from typing import Tuple

from PIL import Image, ImageDraw

from core.dependencies.utils.image.mask_generators.mask_generators import MaskGenerator


class BoxMaskGenerator(MaskGenerator):
    output_size: Tuple[int, int]
    mask_size: Tuple[int, int]
    center: Tuple[int, int]

    def generate(self) -> Image.Image:
        if self.mask_size[0] == 0 and self.mask_size[1] == 0:
            raise ValueError("Mask size cannot be (0, 0)")
        mask = Image.new('L', self.output_size, 0)
        draw = ImageDraw.Draw(mask)
        corner_up_left = (self.center[0] - self.mask_size[0]//2, self.center[1] - self.mask_size[1]//2)
        corner_down_right = (self.center[0] + self.mask_size[0]//2, self.center[1] + self.mask_size[1]//2)
        draw.rectangle((corner_up_left, corner_down_right), fill=255)
        return mask
