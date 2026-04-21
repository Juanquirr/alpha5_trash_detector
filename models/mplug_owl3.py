# venv: .venv-compat (transformers 4.46.x)
# Uses trust_remote_code. ~14GB for 7B variant.
from .base import BaseVLM


class MplugOwl3(BaseVLM):
    name = "mplug_owl3"
    variant = "mPLUG/mPLUG-Owl3-7B-240728"

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.variant, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.variant,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()
        self.processor = self.model.init_processor(self.tokenizer)

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": f"<|image|>\n{prompt}"}]

        inputs = self.processor(messages, images=[image], videos=None)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
