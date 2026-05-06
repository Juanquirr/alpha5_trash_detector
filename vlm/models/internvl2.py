# venv: .transformers-4.46-venv (transformers 4.46.x)
#
# InternVL2-2B: InternViT-300M vision encoder + InternLM2-1.8B language model.
# Consistently top-ranked in 2B-class benchmarks (MMBench, POPE, MME).
# Uses trust_remote_code=True — model code downloaded from HuggingFace Hub.
# Image preprocessing done with PIL + numpy (no torchvision dependency).
#
# Variant options:
#   OpenGVLab/InternVL2-1B   (~2GB)
#   OpenGVLab/InternVL2-2B   (~4GB)  ← active
#   OpenGVLab/InternVL2_5-2B (~4GB)  (newer, same API)
import numpy as np
from .base import BaseVLM

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE    = 448


def _preprocess(image_path: str):
    """Preprocess image to InternVL2's expected format without torchvision."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize(
        (_INPUT_SIZE, _INPUT_SIZE), Image.BICUBIC
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)  # HWC → CHW, shape (3, 448, 448)


class InternVL2(BaseVLM):
    name = "internvl2"
    variant = "OpenGVLab/InternVL2-2B"

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.variant,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            self.variant,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        import torch

        arr = _preprocess(image_path)
        pixel_values = (
            torch.tensor(arr, dtype=torch.bfloat16)
            .unsqueeze(0)           # (1, 3, 448, 448)
            .to(self.device)
        )

        generation_config = dict(max_new_tokens=200, do_sample=False)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )
        return response if isinstance(response, str) else response[0]
