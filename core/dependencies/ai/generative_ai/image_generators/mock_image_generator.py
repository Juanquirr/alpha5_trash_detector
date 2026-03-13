from PIL import Image

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class MockImageGenerator(ImageGenerator):
    route : str

    def generate(self, prompt, negative_prompt) -> Image.Image:
        return Image.open(self.route)