import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path

CLASSES = [
    "plastic bottle",
    "glass",
    "can",
    "plastic bag",
    "plastic wrapper",
    "trash pile",
    "trash",
]

DETECTION_PROMPT = (
    "Look at this image and determine if there is any garbage, litter, or waste visible.\n\n"
    "If you see garbage: respond with YES followed by a colon and the applicable categories "
    "from this list (comma-separated): plastic bottle, glass, can, plastic bag, plastic wrapper, trash pile, trash\n"
    "If there is no garbage: respond with only NO\n\n"
    "Examples:\n"
    "YES: plastic bottle, can\n"
    "YES: trash pile\n"
    "NO"
)


def parse_response(response: str) -> tuple[bool, list[str]]:
    """Parse YES/NO and class labels from model response.

    Robust: scans for class names anywhere in the text, so partial
    or reformatted responses still yield correct classes.
    """
    text = response.strip()
    detected = text.upper().startswith("YES")
    classes = []
    if detected:
        lower = text.lower()
        classes = [c for c in CLASSES if c in lower]
    return detected, classes


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
        t0 = time.perf_counter()
        response = self.describe(image_path, DETECTION_PROMPT)
        elapsed = round(time.perf_counter() - t0, 3)

        detected, classes = parse_response(response)

        vram_mb = 0
        if self.device == "cuda" and torch.cuda.is_available():
            vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            torch.cuda.reset_peak_memory_stats()

        return {
            "image": Path(image_path).name,
            "model": self.name,
            "variant": self.variant,
            "prompt": DETECTION_PROMPT,
            "response": response.strip(),
            "garbage_detected": detected,
            "classes_detected": ", ".join(classes),
            "inference_s": elapsed,
            "vram_mb": vram_mb,
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
