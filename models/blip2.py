# venv: .transformers-4.46-venv (transformers 4.46.x)
#
# Default: blip2-flan-t5-xl (~8GB) — Flan-T5 backbone, encoder-decoder,
# good instruction following. Output contains only generated tokens (no trim needed).
#
# Alternative: Salesforce/blip2-opt-2.7b (~6GB) — OPT backbone, much worse at
# following instructions. If used, output_ids includes the full prompt — needs trimming.
from .base import BaseVLM

_OPT_VARIANTS = {"Salesforce/blip2-opt-2.7b", "Salesforce/blip2-opt-6.7b"}


class BLIP2(BaseVLM):
    name = "blip2"
    variant = "Salesforce/blip2-flan-t5-xl"

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

        # Move to device without casting integer tensors to float16
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        # OPT models include prompt tokens in output — trim them
        if self.variant in _OPT_VARIANTS:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
