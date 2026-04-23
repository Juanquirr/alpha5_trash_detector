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
    "Examine this image carefully and describe what you see, paying attention to any waste, "
    "litter, or garbage on the ground, surfaces, or environment.\n\n"
    "Use the following class definitions to identify waste:\n"
    "- plastic bottle: any plastic CONTAINER with a visible CAP or LID (bottles, jugs, flasks).\n"
    "- glass: glass BOTTLES distinguishable by a bottle NECK shape and glass appearance "
    "(beer, wine, spirits). Does NOT include glass jars.\n"
    "- can: any metal beverage or food can — whole, crushed, or deformed — clearly "
    "identifiable as a can by its cylindrical shape and metallic surface.\n"
    "- plastic bag: clearly a BAG (grocery, garbage, zip-lock). Distinguished from wrappers "
    "by its larger size and bag-like dimensions.\n"
    "- metal scrap: small metal/aluminium items that can be litter — tuna cans, spray cans, "
    "aluminium foil, small metal pieces. NOT structural metal like bars, sheets, or planks "
    "(those are trash).\n"
    "- plastic wrapper: small plastic wrapping — snack bags, candy/chocolate wrappers, "
    "cling film. Smaller and flatter than a plastic bag.\n"
    "- trash pile: an ACCUMULATION of mixed garbage where individual items may or may not "
    "be distinguishable. Must be a visible pile or heap of waste.\n"
    "- trash: any other waste that does not fit the above categories. A catch-all for "
    "unclassifiable litter, including structural metal debris.\n\n"
    "After your description, write exactly one of:\n"
    "DETECTED: <comma-separated classes present from the list above>\n"
    "CLEAN\n\n"
    "Example:\n"
    "The ground shows a crushed plastic bottle near a crumpled snack wrapper. "
    "A small pile of mixed rubbish is visible in the corner.\n"
    "DETECTED: plastic bottle, plastic wrapper, trash pile"
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
