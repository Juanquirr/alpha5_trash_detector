from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import numpy as np
from PIL import Image
import os

from core.dependencies.ai.generative_ai.image_generators.flux_image_generator import FluxImageGenerator
from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator


def read_prompts_from_file(file_path):
    """
    Reads a text file containing prompts separated by new lines
    and returns a list of prompts.

    :param file_path: Path to the .txt file
    :return: List of prompt strings
    """
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # skip empty lines
                prompts.append(stripped_line)
    return prompts


# Example usage:
# prompts_list = read_prompts_from_file('prompts.txt')
# print(prompts_list)

"""

    image_paths = [os.path.join(folder_path, f)
                   for f in sorted(os.listdir(folder_path))
                   if f.startswith(prefix)]

    if not image_paths:
        return 0.0

    process_image = lambda p: np.array(Image.open(p).convert("RGB")) / 255.0

"""

import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image, size=(299, 299)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

sd3_images = list(map(lambda x: preprocess_image(torch.l(Image.open(x))), [os.path.join("./model_comparison_fid", f)
               for f in sorted(os.listdir("./model_comparison_fid"))
               if f.startswith("sd3")]))
flux_images = list(map(lambda x: preprocess_image(torch.from_numpy(Image.open(x))), [os.path.join("./model_comparison_fid", f)
                                                for f in sorted(os.listdir("./model_comparison_fid"))
                                                if f.startswith("flux")]))

from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True)
fid.update(training_real, real=True)
fid.update(training_fake, real=False)

print(f"FID: {float(fid.compute())}")