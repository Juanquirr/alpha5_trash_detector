import uuid

from PIL import Image

from core.dependencies.ai.generative_ai.image_generators.comfy_image_generator import ComfyuiImageGenerator
from core.dependencies.ai.generative_ai.image_generators.comfy_workflows import create_prompt_for_inpainting
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter


class ComfyImageInpainter(ImageInpainter):
    server_address : str = "comfyui.autoescuelaseco.cloud"
    client_id : str = str(uuid.uuid4())

    def inpaint(self, original_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
        return ComfyuiImageGenerator(server_address=self.server_address, client_id=self.client_id, workflow=lambda prompt, negative_prompt : create_prompt_for_inpainting(
            original_image,
            mask_image,
            prompt
        )).generate(prompt, "")