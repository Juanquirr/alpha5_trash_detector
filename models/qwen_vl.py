# venv: .venv (transformers 5.x)
# Uses Qwen2.5-VL. Model ID for Qwen3-VL if available: replace variant below.
# Variant options: Qwen/Qwen2.5-VL-3B-Instruct (~6GB)
#                  Qwen/Qwen2.5-VL-7B-Instruct (~14GB)
from .base import BaseVLM


class QwenVL(BaseVLM):
    name = "qwen_vl"
    variant = "Qwen/Qwen2.5-VL-3B-Instruct"

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
