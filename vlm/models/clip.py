# venv: .transformers-4.46-venv (transformers 4.46.x)
# CLIP is different: zero-shot classification, no text generation.
# Returns top matching classes from CLASSES list + "no garbage".
# garbage_detected = True if any class score >= THRESHOLD.
import time
import torch
from pathlib import Path

from .base import BaseVLM, DEFAULT_CLASSES


# Bare labels used in output and by parse_clip_score — must match POPE question cls field.
_LABELS = DEFAULT_CLASSES + ["no garbage"]

# Richer descriptions fed to the CLIP text encoder — better semantic signal than bare names.
# CLIP zero-shot quality depends on natural-language prompt quality.
_TEXTS = [
    "a floating rigid container such as a plastic or glass bottle",
    "floating plastic material such as a plastic bag or film sheet",
    "floating metal waste such as an aluminium can or crumpled foil",
    "floating white polystyrene foam block or cup",
    "a small floating plastic fragment or bottle cap",
    "a cluster of floating mixed marine waste",
    "floating unidentified debris",
    "no garbage, clean water surface",
]

THRESHOLD = 0.25  # softmax score to consider a class present


class CLIP(BaseVLM):
    name = "clip"
    variant = "openai/clip-vit-large-patch14"

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.variant)
        self.model = CLIPModel.from_pretrained(self.variant).to(self.device)
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        """Returns scored candidates as text (CLIP does not generate free text)."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=_TEXTS, images=image, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        scored = sorted(zip(_LABELS, probs.tolist()), key=lambda x: -x[1])
        return " | ".join(f"{c}: {s:.2f}" for c, s in scored)

    def detect_garbage(self, image_path: str, mode: str = "text") -> dict:
        """Override: CLIP uses zero-shot classification; mode param is ignored."""
        from PIL import Image

        is_cuda = self.device == "cuda" and torch.cuda.is_available()

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=_TEXTS, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - t0, 3)

        scored = list(zip(_LABELS, probs.tolist()))
        garbage_classes = [
            c for c, s in scored
            if c != "no garbage" and s >= THRESHOLD
        ]
        detected = len(garbage_classes) > 0

        response = " | ".join(
            f"{c}: {s:.3f}" for c, s in sorted(scored, key=lambda x: -x[1])
        )

        vram_mb = 0
        if is_cuda:
            vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

        return {
            "image": Path(image_path).name,
            "model": self.name,
            "variant": self.variant,
            "prompt": f"zero-shot classification, threshold={THRESHOLD}",
            "response": response,
            "garbage_detected": detected,
            "classes_detected": ", ".join(garbage_classes),
            "inference_s": elapsed,
            "vram_mb": vram_mb,
        }
