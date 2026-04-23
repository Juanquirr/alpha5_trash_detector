# venv: .venv-compat (transformers 4.46.x)
# Requires HuggingFace token: set HF_TOKEN env var or run `huggingface-cli login`
# PaliGemma is a prefix-completion model, not chat. Prompt is the prefix.
from .base import BaseVLM, DETECTION_PROMPT


# PaliGemma uses shorter, prefix-style prompts better than long instructions.
# Override with a compact prompt.
# PaliGemma is a prefix-completion model — short prompts work better.
# Describe-first still applies: ask for scene description then classification.
_PALI_PROMPT = (
    "Describe visible waste. Classes: "
    "plastic bottle=plastic container with cap; "
    "glass=glass bottle with neck shape; "
    "can=metal cylindrical can; "
    "plastic bag=large bag shape; "
    "metal scrap=small metal/aluminium litter; "
    "plastic wrapper=small snack/candy wrapping; "
    "trash pile=accumulated heap of mixed waste; "
    "trash=other unclassifiable waste. "
    "End with DETECTED: <classes> or CLEAN."
)


class PaliGemma(BaseVLM):
    name = "paligemma"
    variant = "google/paligemma-3b-mix-448"

    def load(self) -> None:
        import torch
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # PaliGemma uses its own prompt, not the shared one
        inputs = self.processor(
            text=_PALI_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
