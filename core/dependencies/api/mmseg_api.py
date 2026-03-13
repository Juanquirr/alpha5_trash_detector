import hashlib
from typing import List, Any, Tuple, Optional
from io import BytesIO
from PIL import Image
import base64
import requests
import os

from pydantic import BaseModel


class MMSegAPI(BaseModel):
    url : str
    cache_dir : str = "./segmentation_cache"

    @staticmethod
    def to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def segment_image(self, image: Image.Image) -> Image.Image:
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if self.is_in_cache(image):
            return self.recover_from_cache(image)
        payload = self.create_payload([self.to_base64(image)])
        response = requests.post(self.url + "/inference/images", json=payload)
        b64_img = response.json()["segmented_images"][0]
        final_image = Image.open(BytesIO(base64.b64decode(b64_img)))
        self.save_in_cache(image, final_image)
        return final_image

    def is_in_cache(self, image: Image.Image):
        hash = self.get_hash(image)
        if hash + ".png" in os.listdir(self.cache_dir):
            return True
        return False

    def recover_from_cache(self, image : Image.Image):
        hash = self.get_hash(image)
        return Image.open(self.cache_dir + "/" + hash + ".png")

    def get_hash(self, image: Image.Image) -> str:
        return hashlib.sha1(image.tobytes()).hexdigest()

    def save_in_cache(self, original_image : Image.Image, segmented_image : Image.Image):
        hash = self.get_hash(original_image)
        segmented_image.save(self.cache_dir + "/" + hash + ".png")

    @staticmethod
    def create_payload(images: List[str]) -> dict:
        return {
            "config_engine": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py",
            "framework": "pytorch",
            "checkpoint": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/val_da_best_mIoU_iter_16000.pth",
            "output_prefix": "result-",
            "device": "cuda:0",
            "opacity": 1,
            "title": "result",
            "labels": "False",
            "custom_ops": None,
            "input_size": None,
            "precision": None,
            "images_base64": images
        }

    def get_inference_color(self, category: str) -> Tuple[int, int, int]:
        response = requests.get(self.url + "/inference_colors/")
        response.raise_for_status()
        return tuple(response.json()[category])

