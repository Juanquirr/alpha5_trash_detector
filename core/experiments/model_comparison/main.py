import os

from dotenv import load_dotenv

from core.dependencies.ai.generative_ai.image_generators.flux_image_generator import FluxImageGenerator
from core.dependencies.ai.generative_ai.image_generators.stable_diffusion_image_generator import \
    StableDiffusionImageGenerator

import os
import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

load_dotenv()

"""
ChatGPT prompt:

Generate 40 text-to-image prompts in the form of a Python list. Each prompt should describe a maritime environment featuring different types of boats: Cargo ships, Kayaks, Fishing boats, Yachts, Inflatable boats, Cruise ships, Open motorboats, and Sailboats.

Divide the prompts into four groups of 10 based on complexity:

Simple situations: One boat of a single type in a basic maritime setting.

Moderately complex: Multiple boats of the same type interacting or sharing space.

Mixed-type scenarios: Several boats of different types in a shared environment.

Highly complex scenes: Busy harbors, regattas, or chaotic rescue scenes with many boats of various types.

Include environmental context (weather, lighting, water conditions) to make the scenes more vivid and suitable for a diffusion-based text-to-image model.
"""

prompts = [
    # Simple Situations (1–10)
    "A lone cargo ship sailing across a calm sea under a clear blue sky at midday",
    "A single kayak drifting peacefully on a still lake surrounded by misty mountains at sunrise",
    "A fishing boat anchored in shallow coastal waters during a golden hour sunset, with seagulls flying overhead",
    "A luxury yacht anchored near a tropical island with turquoise water and a partly cloudy sky",
    "An inflatable boat floating near a quiet riverbank with dense green forest under soft morning light",
    "A cruise ship cruising along the open ocean beneath a vivid sunset sky with gentle waves",
    "A small open motorboat gliding through a glassy fjord, surrounded by steep cliffs in overcast weather",
    "A white sailboat with sails down resting on calm coastal waters during twilight with stars beginning to appear",
    "A cargo ship docked at an industrial port on a rainy afternoon, water rippling from gentle rain",
    "A red kayak paddling near icebergs in the Arctic under a pale blue sky with soft polar light",

    # Moderately Complex (11–20)
    "Three cargo ships lined up near a harbor under cloudy skies with calm water reflecting their hulls",
    "A group of kayakers navigating through a river with mild rapids on a sunny spring day",
    "Several fishing boats tied together in a bay, nets drying in the wind with a partly cloudy sky above",
    "Multiple yachts sailing in formation during a coastal cruise under a bright, breezy midday sun",
    "Four inflatable boats drifting near a beach during training exercises, with light chop on the sea and clear skies",
    "Two cruise ships passing each other in open ocean under dramatic golden sunset lighting",
    "Several open motorboats racing across a lake with white spray and morning fog lifting off the water",
    "Three sailboats circling a small island on a sunny afternoon with moderate wind and clear water",
    "Two cargo ships slowly navigating through a narrow canal under low grey clouds and scattered rain",
    "Five kayaks exploring a mangrove estuary during golden hour with still water reflecting the trees",

    # Mixed-Type Scenarios (21–30)
    "A cruise ship passing near a group of kayaks in a tropical bay under a bright blue sky",
    "A fishing boat and a sailboat sharing a calm harbor at sunset with light waves lapping at the dock",
    "A luxury yacht surrounded by small inflatable boats and jet skis near a busy beach on a sunny afternoon",
    "A cargo ship being escorted by two motorboats in foggy conditions near a stormy coast",
    "Several sailboats and kayaks scattered across a mountain lake on a misty morning",
    "A fishing boat unloading next to a docked cargo ship with overcast skies and seabirds flying around",
    "Open motorboats and yachts floating near each other at a party cove under a warm sunset",
    "An inflatable boat approaching a sailboat anchored offshore during a clear afternoon",
    "A cruise ship docked beside a cargo ship in a busy international port under partly cloudy skies",
    "A group of kayaks paddling past a fishing boat with dolphins swimming nearby under sunny weather",

    # Highly Complex Scenes (31–40)
    "A bustling harbor at sunset with cargo ships, sailboats, fishing boats, and motorboats navigating around each other",
    "A regatta with dozens of sailboats racing under strong wind conditions and bright midday sunlight",
    "An emergency sea rescue with inflatable boats, helicopters, and a yacht in distress under stormy skies",
    "A maritime festival featuring cruise ships, fishing boats, and kayaks decorated with colorful lights at night",
    "A coastal storm evacuation scene with various boats rushing through choppy waters under dark thunderclouds",
    "A busy dockyard with cargo ships unloading, fishing boats refueling, and sailboats preparing to depart at dawn",
    "A nighttime harbor scene illuminated by moonlight with yachts, motorboats, and cargo ships gently bobbing on dark water",
    "An ocean research mission with sailboats, inflatable boats, and a large support ship under cloudy skies",
    "A summer boating event with dozens of open motorboats, kayaks, and yachts in crystal clear water under a hot sun",
    "A chaotic regatta finale with sailboats converging near the finish line, inflatable boats providing support, and storm clouds approaching"
]


def run_generation():
    for x in range(0, len(prompts)):
        sd35_large_generator = StableDiffusionImageGenerator(api_key=os.getenv("SD_API_KEY", ""),
                                                         prompt=prompts[x]
                                                         )
        flux11_dev_generator = FluxImageGenerator(api_key=os.getenv("FLUX_API_KEY", ""),
                                                             prompt=prompts[x]
                                                             )
        img1 = sd35_large_generator.generate()
        img2 = flux11_dev_generator.generate()
        img1.save("model_comparison/flux-{}.png".format(x))
        img2.save("model_comparison/sd35-{}.png".format(x))

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def clip_score_from_folder(folder_path, prefix, prompts):
    """
    Loads images with a given prefix from a folder (without resizing),
    normalizes them, and computes the CLIP score.

    Args:
        folder_path (str): Path to folder with images.
        prefix (str): Prefix to filter image filenames.
        prompts (List[str]): Text prompts to compare against.

    Returns:
        float: CLIP score.
    """
    image_paths = [os.path.join(folder_path, f)
                   for f in sorted(os.listdir(folder_path))
                   if f.startswith(prefix)]

    if not image_paths:
        return 0.0

    process_image = lambda p: np.array(Image.open(p).convert("RGB")) / 255.0
    images = np.stack(list(map(process_image, image_paths)))

    images_int = (images * 255).astype("uint8")
    images_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2)

    score = clip_score_fn(images_tensor, prompts).detach()
    return round(float(score), 4)

print(clip_score_from_folder("./model_comparison", "flux", prompts))