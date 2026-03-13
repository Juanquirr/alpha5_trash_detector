import os

from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from core.dependencies.ai.discriminative_ai.background_removers.stable_diffusion_background_remover import \
    StableDiffusionBackgroundRemover


def test_stable_diffusion_background_remover():
    load_dotenv()
    bg_remover = StableDiffusionBackgroundRemover(api_key=os.getenv("SD_API_KEY"))
    img = bg_remover.remove(Image.open("image_1.jpeg"))
    plt.imshow(img)
    plt.show()