# venv: .transformers-4.46-venv (transformers 4.46.x)
# Uses IDEFICS2 (successor to IDEFICS). ~8GB.
from .base import BaseVLM


class IDEFICS(BaseVLM):
    name = "idefics"
    variant = "HuggingFaceM4/idefics2-8b"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

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
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
