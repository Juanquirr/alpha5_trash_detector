# venv: .transformers-5.X-venv (transformers 5.x)
# SmolVLM-500M: lighter variant of SmolVLM-2B. Same API, ~4x smaller.
# Useful for speed/accuracy tradeoff comparison.
from .base import BaseVLM


class SmolVLM500M(BaseVLM):
    name = "smolvlm_500m"
    variant = "HuggingFaceTB/SmolVLM-500M-Instruct"

    def load(self) -> None:
        from PIL import Image  # noqa: F401 — ensure Pillow available early
        import torch
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText as AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoProcessor, AutoModelForVision2Seq

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.variant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True)
