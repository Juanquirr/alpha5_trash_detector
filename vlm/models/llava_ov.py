# venv: .transformers-5.X-venv (transformers 5.x)
#
# LLaVA-OneVision-0.5B: Qwen2-0.5B backbone + SigLIP vision encoder.
# Extremely small (0.5B) yet competitive on visual reasoning tasks.
# Uses LlavaOnevisionForConditionalGeneration (added in transformers 4.45).
#
# Variant options:
#   lmms-lab/llava-onevision-qwen2-0.5b-ov   (~1GB)  ← active (general)
#   lmms-lab/llava-onevision-qwen2-0.5b-si   (~1GB)  (single-image fine-tuned)
#   lmms-lab/llava-onevision-qwen2-7b-ov     (~14GB) (7B variant, too large)
from .base import BaseVLM


class LlavaOV(BaseVLM):
    name = "llava_ov"
    variant = "lmms-lab/llava-onevision-qwen2-0.5b-ov"

    def load(self) -> None:
        import torch
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.float16,
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
            images=[image], text=[text], return_tensors="pt"
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True)
