import json
import re
import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path

DEFAULT_CLASSES = [
    "container",
    "plastic",
    "metal",
    "polystyrene",
    "plastic_fragment",
    "trash_pile",
    "trash",
]

# ── Prompts ────────────────────────────────────────────────────────────────────

DETECTION_PROMPT = (
    "Examine this image carefully and describe what you see, paying attention to any waste, "
    "litter, or garbage floating on or near water surfaces.\n\n"
    "Use the following class definitions to identify waste:\n"
    "- container: rigid non-metal container with elongated or cylindrical shape — plastic "
    "or glass bottles, jars, rigid cups. Identified by shape, not material.\n"
    "- plastic: flat, amorphous, flexible, translucent plastic material — bags, film, "
    "soft wrappers, floating soft plastic sheets.\n"
    "- metal: any item with a specular METALLIC REFLECTION — cans, aluminium foil, "
    "metal scrap. Identified by its shiny metallic surface.\n"
    "- polystyrene: white opaque MATTE foam material — EPS foam blocks, polystyrene cups "
    "or plates, white cork-like foam debris.\n"
    "- plastic fragment: small, compact, rigid 3D plastic piece — bottle caps, broken "
    "plastic fragments, plastic cutlery, straws.\n"
    "- trash pile: dense CLUSTER or ACCUMULATION of multiple waste objects forming a heap "
    "of mixed garbage where individual items may be indistinguishable.\n"
    "- trash: single unclassifiable waste item that cannot be identified as any of the "
    "above — pallets, pellets, fauna, glass shards, other debris.\n\n"
    "After your description, write exactly one of:\n"
    "DETECTED: <comma-separated classes present from the list above>\n"
    "CLEAN\n\n"
    "Example:\n"
    "The water surface shows a rigid container floating near a sheet of translucent plastic film. "
    "A dense cluster of mixed refuse is visible in the corner.\n"
    "DETECTED: container, plastic, trash pile"
)

DETECTION_PROMPT_JSON = (
    "Examine this image for waste or litter floating on or near water.\n\n"
    "Return ONLY a JSON object. Each value is the count of that item visible (0 if absent):\n\n"
    "{\n"
    '  "container": 0,\n'
    '  "plastic": 0,\n'
    '  "metal": 0,\n'
    '  "polystyrene": 0,\n'
    '  "plastic_fragment": 0,\n'
    '  "trash_pile": 0,\n'
    '  "trash": 0\n'
    "}\n\n"
    "Definitions:\n"
    "- container: rigid non-metal container with elongated/cylindrical shape — bottles, jars, rigid cups\n"
    "- plastic: flat, flexible, translucent plastic — bags, film, soft wrappers, floating plastic sheets\n"
    "- metal: specular metallic reflection — cans, aluminium foil, metal scrap\n"
    "- polystyrene: white opaque matte foam — EPS foam blocks, polystyrene cups or plates\n"
    "- plastic_fragment: small compact rigid plastic — bottle caps, broken fragments, cutlery, straws\n"
    "- trash_pile: dense cluster of multiple waste objects forming a mixed heap\n"
    "- trash: single unclassifiable item — pallets, pellets, fauna, glass shards, other\n\n"
    "Output only the JSON object, no explanation."
)

# ── Parsers ────────────────────────────────────────────────────────────────────

_JSON_KEY_TO_CLASS = {
    "container":        "container",
    "plastic":          "plastic",
    "metal":            "metal",
    "polystyrene":      "polystyrene",
    "plastic_fragment": "plastic_fragment",
    "trash_pile":       "trash_pile",
    "trash":            "trash",
}


def _extract_classes(text: str) -> list[str]:
    """Longest-match-first with consume: prevents 'trash' matching inside 'trash pile'."""
    lower = text.lower()
    found = []
    for cls in sorted(DEFAULT_CLASSES, key=len, reverse=True):
        # Match both underscore and space variants (VLM responses use spaces)
        pattern = r"\b" + re.escape(cls).replace(r"\_", r"[_ ]") + r"\b"
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
