import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path

CLASSES = [
    "plastic bottle",
    "glass",
    "can",
    "plastic bag",
    "metal scrap",
    "plastic wrapper",
    "trash pile",
    "trash",
]

DETECTION_PROMPT = (
    "Describe what you see in this image in detail, focusing on any objects on the ground, "
    "surfaces, or environment. Note materials, conditions, and any signs of waste or cleanliness.\n\n"
    "After your description, conclude with exactly one of these two lines:\n"
    "DETECTED: <comma-separated types from this list: "
    "plastic bottle, glass, can, plastic bag, plastic wrapper, trash pile, trash>\n"
    "CLEAN\n\n"
    "Example responses:\n"
    "The image shows a park path with several crushed plastic bottles and an aluminum can "
    "scattered on the grass near a bench.\nDETECTED: plastic bottle, can\n\n"
    "The image shows a clean sidewalk with no visible litter.\nCLEAN"
)


def parse_response(response: str) -> tuple[bool, list[str]]:
    """Parse DETECTED/CLEAN label and class names from model response.

    Strategy:
    1. Look for DETECTED: or CLEAN at end of response (expected format).
    2. Fallback: scan full text for class names if label missing.
    """
    text = response.strip()
    upper = text.upper()

    # Primary: structured label at end
    if "DETECTED:" in upper:
        detected = True
        # Extract everything after last DETECTED:
        after = text[upper.rfind("DETECTED:") + len("DETECTED:"):].strip()
        lower = after.lower()
        classes = [c for c in CLASSES if c in lower]
        # Fallback to full text scan if nothing matched after label
        if not classes:
            classes = [c for c in CLASSES if c in text.lower()]
        return detected, classes

    if upper.endswith("CLEAN") or "\nCLEAN" in upper:
        return False, []

    # Fallback: no structured label — scan full response for class names
    lower = text.lower()
    classes = [c for c in CLASSES if c in lower]
    detected = len(classes) > 0
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
