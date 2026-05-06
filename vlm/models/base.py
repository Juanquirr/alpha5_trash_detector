import json
import re
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

# ── Prompts ────────────────────────────────────────────────────────────────────

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

DETECTION_PROMPT_JSON = (
    "Examine this image for waste or litter.\n\n"
    "Return ONLY a JSON object. Each value is the count of that item visible (0 if absent):\n\n"
    "{\n"
    '  "plastic_bottle": 0,\n'
    '  "glass": 0,\n'
    '  "can": 0,\n'
    '  "plastic_bag": 0,\n'
    '  "metal_scrap": 0,\n'
    '  "plastic_wrapper": 0,\n'
    '  "trash_pile": 0,\n'
    '  "trash": 0\n'
    "}\n\n"
    "Definitions:\n"
    "- plastic_bottle: plastic CONTAINER with a visible CAP or LID (bottles, jugs, flasks)\n"
    "- glass: glass BOTTLE with NECK shape (beer, wine, spirits — NOT jars)\n"
    "- can: metal beverage/food can — cylindrical, whole or crushed\n"
    "- plastic_bag: BAG shape — grocery, garbage, zip-lock\n"
    "- metal_scrap: small metal/aluminium litter — tuna cans, foil, spray cans\n"
    "- plastic_wrapper: small snack/candy wrapping — smaller and flatter than a bag\n"
    "- trash_pile: ACCUMULATION of mixed garbage — visible pile or heap\n"
    "- trash: any other unclassifiable waste\n\n"
    "Output only the JSON object, no explanation."
)

# ── Parsers ────────────────────────────────────────────────────────────────────

_JSON_KEY_TO_CLASS = {
    "plastic_bottle":  "plastic bottle",
    "glass":           "glass",
    "can":             "can",
    "plastic_bag":     "plastic bag",
    "metal_scrap":     "metal scrap",
    "plastic_wrapper": "plastic wrapper",
    "trash_pile":      "trash pile",
    "trash":           "trash",
}


def _extract_classes(text: str) -> list[str]:
    """Longest-match-first with consume: prevents 'trash' matching inside 'trash pile'."""
    lower = text.lower()
    found = []
    for cls in sorted(CLASSES, key=len, reverse=True):
        pattern = r"\b" + re.escape(cls) + r"\b"
        if re.search(pattern, lower):
            found.append(cls)
            lower = re.sub(pattern, " " * len(cls), lower)
    return found


def parse_response(response: str) -> tuple[bool, list[str]]:
    text = response.strip()
    upper = text.upper()

    if "DETECTED:" in upper:
        after = text[upper.rfind("DETECTED:") + len("DETECTED:"):].strip()
        classes = _extract_classes(after) or _extract_classes(text)
        return True, classes

    if upper.endswith("CLEAN") or "\nCLEAN" in upper:
        return False, []

    classes = _extract_classes(text)
    return len(classes) > 0, classes


def parse_json_response(response: str) -> tuple[bool, list[str]]:
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if not match:
        return False, []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return False, []
    classes = [
        cls
        for key, cls in _JSON_KEY_TO_CLASS.items()
        if int(data.get(key, 0) or 0) > 0
    ]
    return len(classes) > 0, classes


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseVLM(ABC):
    name: str = ""
    variant: str = ""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def describe(self, image_path: str, prompt: str) -> str: ...

    def _get_prompt(self, mode: str) -> str:
        return DETECTION_PROMPT_JSON if mode == "json" else DETECTION_PROMPT

    def detect_garbage(self, image_path: str, mode: str = "text") -> dict:
        prompt = self._get_prompt(mode)
        is_cuda = self.device == "cuda" and torch.cuda.is_available()

        # Reset peak BEFORE inference so measurement captures only this forward pass,
        # not model loading or previous images.
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        response = self.describe(image_path, prompt)
        # Synchronize before stopping timer: GPU ops are async, without this
        # perf_counter() returns before the kernel finishes.
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - t0, 3)

        if mode == "json":
            detected, classes = parse_json_response(response)
        else:
            detected, classes = parse_response(response)

        vram_mb = 0
        if is_cuda:
            vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

        return {
            "image": Path(image_path).name,
            "model": self.name,
            "variant": self.variant,
            "prompt": prompt,
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
