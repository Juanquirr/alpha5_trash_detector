from typing import Tuple, Optional

from PIL import Image

from core.dependencies.utils.image.image_editors.image_editor import ImageEditor


class PixelTransparencyThresholdCleanEditor(ImageEditor):
    threshold : float
    clean_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0)

    def edit(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA format")
        cleaned_image = Image.new("RGBA", image.size)
        pixels = image.load()
        cleaned_pixels = cleaned_image.load()

        for y in range(image.height):
            for x in range(image.width):
                r, g, b, alpha = pixels[x, y]

                # Check if the alpha value is below the threshold
                if alpha < self.threshold:
                    # Make the pixel fully transparent
                    cleaned_pixels[x, y] = self.clean_color
                else:
                    # Keep the original pixel
                    cleaned_pixels[x, y] = (r, g, b, alpha)

        return cleaned_image
