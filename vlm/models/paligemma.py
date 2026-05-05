# venv: .transformers-4.46-venv (transformers 4.46.x)
# Requires HuggingFace token: set HF_TOKEN env var or run `huggingface-cli login`
# PaliGemma is a prefix-completion model, not chat. Uses shorter prompts than the default.
from .base import BaseVLM

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

_PALI_PROMPT_JSON = (
    "Examine for waste. Return only JSON with counts (0 if absent): "
    '{"plastic_bottle":0,"glass":0,"can":0,"plastic_bag":0,'
    '"metal_scrap":0,"plastic_wrapper":0,"trash_pile":0,"trash":0}'
)


class PaliGemma(BaseVLM):
    name = "paligemma"
    variant = "google/paligemma-3b-mix-448"

    def _get_prompt(self, mode: str) -> str:
        return _PALI_PROMPT_JSON if mode == "json" else _PALI_PROMPT

    def load(self) -> None:
        import os
        import torch
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self._torch = torch
        token = os.environ.get("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(self.variant, token=token)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.variant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            token=token,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
