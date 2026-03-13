from enum import IntEnum
from typing import Optional, Callable, Tuple

from PIL import Image
from pydantic import ConfigDict

from core.dependencies.utils.image.mask_generators.mask_generators import MaskGenerator


class AlphaMaskGenerator(MaskGenerator):
    class Type(IntEnum):
        inside = 1
        outside = 2
        border_inside = 3
        border_outside = 4

    alpha_image : Image.Image
    type : Type
    strength: int = 255
    border_width : int = 2

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def generate(self) -> Image.Image:
        self.alpha_image = self.alpha_image.convert("RGBA")
        if self.type == AlphaMaskGenerator.Type.inside:
                return self.__generate_fill_mask_inside()
        if self.type == AlphaMaskGenerator.Type.outside:
                return self.__generate_fill_mask_outside()
        if self.type == AlphaMaskGenerator.Type.border_inside:
            return self.__generate_border_inside_mask()
        if self.type == AlphaMaskGenerator.Type.border_outside:
            return self.__generate_border_outside_mask()


    def __map_pixel(self, image : Image.Image, f : Callable[[Image.Image, Tuple[int, int]], None]) -> Image.Image:
        img = image.copy()
        for x in range(self.alpha_image.size[0]):
            for y in range(self.alpha_image.size[1]):
                f(img, (x, y))
        return img

    def __generate_fill_mask_inside(self) -> Image.Image:
        def filter(image : Image.Image, pos : Tuple[int, int]):
            if self.alpha_image.getpixel(pos)[3] != 0:
                image.putpixel(pos, self.strength)
        return self.__map_pixel(Image.new("L", self.alpha_image.size, 0), filter)

    def __generate_fill_mask_outside(self) -> Image.Image:
        def filter(image : Image.Image, pos : Tuple[int, int]):
            if self.alpha_image.getpixel(pos)[3] == 0:
                image.putpixel(pos, self.strength)
        return self.__map_pixel(Image.new("L", self.alpha_image.size), filter)

    def __generate_border_inside_mask(self) -> Image.Image:
        mask = Image.new("L", self.alpha_image.size, 0)
        if self.border_width is None or self.border_width <= 0:
            return mask  # No border to generate

        def filter(image: Image.Image, pos: Tuple[int, int]):
            x, y = pos
            if self.alpha_image.getpixel((x, y))[3] > 0:  # Only consider non-transparent pixels
                for dx in range(-self.border_width, self.border_width + 1):
                    for dy in range(-self.border_width, self.border_width + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.alpha_image.size[0] and 0 <= ny < self.alpha_image.size[1]:
                            if self.alpha_image.getpixel((nx, ny))[3] == 0:  # Adjacent to transparent pixel
                                image.putpixel((x, y), self.strength)  # Set border pixel value
                                return  # Exit early if the border condition is met

        return self.__map_pixel(mask, filter)

    def __generate_border_outside_mask(self) -> Image.Image:
        mask = Image.new("L", self.alpha_image.size, 0)
        if self.border_width is None or self.border_width <= 0:
            return mask  # No border to generate
        def filter(image: Image.Image, pos: Tuple[int, int]):
            x, y = pos
            if self.border_width is None or self.border_width <= 0: return
            if self.alpha_image.getpixel((x, y))[3] > 0:
                for dx in range(-self.border_width, self.border_width + 1):
                    for dy in range(-self.border_width, self.border_width + 1):
                        if 0 <= x + dx <= self.alpha_image.size[0] and 0 <= y + dy <= self.alpha_image.size[1]:
                            if self.alpha_image.getpixel((x + dx, y + dy))[3] == 0:
                                image.putpixel((x + dx, y + dy), self.strength)
        return self.__map_pixel(mask, filter)
