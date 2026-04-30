from .base import BaseVLM


class Moondream(BaseVLM):
    name = "moondream"
    variant = "vikhyatk/moondream2"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.variant, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.variant,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        encoded = self.model.encode_image(image)
        return self.model.answer_question(encoded, prompt, self.tokenizer)
