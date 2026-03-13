from typing import Optional
from PIL import Image
import requests
from io import BytesIO
import time

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class FluxImageGenerator(ImageGenerator):
    api_key: str
    image_prompt: Optional[str] = None  # base64 image if needed
    width: int = 1024
    height: int = 768
    steps: int = 28
    prompt_upsampling: bool = False
    seed: Optional[int] = None
    guidance: float = 3.0
    safety_tolerance: int = 2
    output_format: str = "png"

    def generate(self, prompt, negative_prompt) -> Image.Image:
        payload = {
            "prompt": prompt,
            "image_prompt": self.image_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "prompt_upsampling": self.prompt_upsampling,
            "seed": self.seed,
            "guidance": self.guidance,
            "safety_tolerance": self.safety_tolerance,
            "output_format": self.output_format,
        }

        response = requests.post(
            "https://api.us1.bfl.ai/v1/flux-dev",
            headers={"Content-Type": "application/json", "x-key": self.api_key},
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Flux API error: {response.status_code} - {response.text}")

        result = response.json()
        polling_url = result.get("polling_url")

        if not polling_url:
            raise Exception("No polling URL returned from FLUX API.")

        # Polling for the result
        for _ in range(10000):  # up to ~30 seconds
            poll_response = requests.get(polling_url, headers={ "x-key": self.api_key})
            if poll_response.status_code == 200:
                poll_data = poll_response.json()
                image_url = poll_data.get("result")
                if image_url:
                    image_resp = requests.get(image_url["sample"])
                    return Image.open(BytesIO(image_resp.content))
            elif poll_response.status_code == 202:
                time.sleep(1)  # still processing
            else:
                raise Exception(f"Polling error: {poll_response.status_code} - {poll_response.text}")

        raise TimeoutError("Image generation timed out.")
