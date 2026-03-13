import os

import matplotlib.pyplot as plt
from PIL import Image
from core.dependencies.ai.generative_ai.image_inpainters.stable_diffusion_image_inpainter import \
    StableDiffusionImageInpainter
from dotenv import load_dotenv

from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator


def test_stable_diffusion_image_inpainter():
    load_dotenv()
    inpainter = StableDiffusionImageInpainter(api_key=os.getenv("SD_API_KEY"), prompt="a boat in the shipyard being repared", negative_prompt="boats")
    mask = AlphaMaskGenerator(type=AlphaMaskGenerator.Type.outside, alpha_image=Image.open("test.png")).generate()
    #image = inpainter.inpaint(original_image=Image.open("../../../../../../data_assets/boats/with_bg/image_1.jpeg"), mask_image=mask)
    #plt.imshow(image)
    #plt.show()