from .base import BaseVLM


class Moondream(BaseVLM):
    name = "moondream"
    variant = "vikhyatk/moondream2"

    def load(self) -> None:
        import moondream as md
        self.model = md.vl(model=self.variant)

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        encoded = self.model.encode_image(image)
        return self.model.query(encoded, prompt)["answer"]
