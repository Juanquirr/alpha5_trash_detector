from pydantic import BaseModel
from PIL import Image
from typing import Dict, Union
import numpy as np
import base64
from io import BytesIO
import requests
from typing import Optional


class DiffuserResult(BaseModel):
    image: Image.Image
    attention_maps: Dict[str, np.ndarray]

    model_config = {
        "arbitrary_types_allowed": True  # 👈 This fixes the schema error
    }

    @classmethod
    def from_api(cls, data: dict) -> "DiffuserResult":
        # Decode image
        img_bytes = base64.b64decode(data["image_base64"])
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Convert attention maps to np.ndarray
        attn_maps = {
            k: np.array(v, dtype=np.float32)
            for k, v in data["attention_maps"].items()
        }

        return cls(image=image, attention_maps=attn_maps)

class GenerationParams(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    height: Optional[int] = None
    width: Optional[int] = None

class DiffusersServiceClient(BaseModel):
    base_url : str

    def generate(
        self,
        params: Union[str, GenerationParams],
        timeout: int = 60
    ) -> DiffuserResult:
        payload = params.dict(exclude_none=True)

        response = requests.post(f"{self.base_url}/generate", json=payload, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")

        return DiffuserResult.from_api(response.json())
