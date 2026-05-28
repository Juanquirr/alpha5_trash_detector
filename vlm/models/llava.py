# venv: .transformers-4.46-venv (transformers 4.46.x)
# LLaVA 1.5 7B. Requires <image> token injected into prompt.
# Variant options:
#   llava-hf/llava-1.5-7b-hf   (~14GB) ← active
#   llava-hf/llava-1.5-13b-hf  (~26GB)
from .base import BaseVLM


class LLaVA(BaseVLM):
    name = "llava"
    variant = "llava-hf/llava-1.5-7b-hf"

    def load(self) -> None:
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # LLaVA 1.5 requires <image> token in prompt
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(
            text=full_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True)
