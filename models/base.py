import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path


class BaseVLM(ABC):
    name: str = ""
    variant: str = ""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""
        ...

    @abstractmethod
    def describe(self, image_path: str, prompt: str) -> str:
        """Return raw text response for given image and prompt."""
        ...

    def detect_garbage(self, image_path: str) -> dict:
        prompt = (
            "Is there garbage, trash, litter, or waste in this image? "
            "Answer YES or NO, then briefly describe what you see."
        )
        t0 = time.perf_counter()
        response = self.describe(image_path, prompt)
        elapsed = round(time.perf_counter() - t0, 3)

        detected = response.strip().upper().startswith("YES")

        vram_mb = 0
        if self.device == "cuda" and torch.cuda.is_available():
            vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            torch.cuda.reset_peak_memory_stats()

        return {
            "image": Path(image_path).name,
            "model": self.name,
            "variant": self.variant,
            "prompt": prompt,
            "response": response.strip(),
            "garbage_detected": detected,
            "inference_s": elapsed,
            "vram_mb": vram_mb,
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
