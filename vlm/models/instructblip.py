# venv: .venv-compat (transformers 4.46.x)
# Variant options: Salesforce/instructblip-vicuna-7b (~14GB)
#                  Salesforce/instructblip-flan-t5-xl (~8GB) — lighter
from .base import BaseVLM


class InstructBLIP(BaseVLM):
    name = "instructblip"
    variant = "Salesforce/instructblip-flan-t5-xl"

    def load(self) -> None:
        import torch
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        self._torch = torch
        self.processor = InstructBlipProcessor.from_pretrained(self.variant)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=5,
            )

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
