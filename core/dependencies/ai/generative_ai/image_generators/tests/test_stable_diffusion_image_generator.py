from matplotlib import pyplot as plt
import os
from dotenv import load_dotenv

from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator


def test_generation():
    load_dotenv()
    generator = StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY"), prompt="A photo of a cargo ship")
    #image = generator.generate()
    #plt.imshow(image)
    #plt.show()