# venv: .transformers-5.X-venv (transformers 5.x)
#
# Qwen3-VL-2B-Instruct: 2B vision-language model from Qwen3 series.
# Uses AutoModelForVision2Seq (generic) instead of series-specific class
# so it works regardless of the exact Transformers class name for Qwen3-VL.
#
# Variant options:
#   Qwen/Qwen3-VL-2B-Instruct  (~4GB)  ← active
#   Qwen/Qwen3-VL-7B-Instruct  (~14GB)
from .base import BaseVLM


class QwenVL2B(BaseVLM):
    name = "qwen_2b"
    variant = "Qwen/Qwen3-VL-2B-Instruct"

    def load(self) -> None:
        import torch
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText as _ModelCls
        except ImportError:
            from transformers import AutoProcessor, AutoModelForVision2Seq as _ModelCls

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        # Cap image resolution globally so both eval and LoRA training stay
        # within VRAM budget. Qwen3-VL tiles images dynamically; without this,
        # large images produce hundreds of vision tokens and OOM on a 32 GiB GPU.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 512 * 28 * 28
        self.model = _ModelCls.from_pretrained(
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
            text=[text], images=[image], return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
