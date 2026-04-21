# venv: .venv-compat (transformers 4.46.x)
# Variant: Salesforce/blip2-opt-2.7b (~6GB) or blip2-flan-t5-xl (~8GB)
from .base import BaseVLM


class BLIP2(BaseVLM):
    name = "blip2"
    variant = "Salesforce/blip2-opt-2.7b"

    def load(self) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, self._torch.float16
        )

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
