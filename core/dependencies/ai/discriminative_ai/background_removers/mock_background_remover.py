from PIL import Image

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover


class MockBackgroundRemover(BackgroundRemover):
    def remove(self, image: Image.Image) -> Image.Image:
        return image