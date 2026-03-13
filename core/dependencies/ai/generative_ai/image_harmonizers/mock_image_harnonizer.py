from PIL  import Image

from core.dependencies.ai.generative_ai.image_harmonizers.image_harmonizer import ImageHarmonizer


class MockImageHarmonizer(ImageHarmonizer):
    def harmonize(self, image : Image.Image, mask: Image.Image) -> Image.Image:
        return image
