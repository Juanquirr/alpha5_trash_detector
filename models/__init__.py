from .smolvlm import SmolVLM
from .moondream import Moondream
from .llava import LLaVA
from .blip2 import BLIP2
from .instructblip import InstructBLIP
from .clip import CLIP
from .paligemma import PaliGemma
from .idefics import IDEFICS
from .mplug_owl3 import MplugOwl3
from .qwen_vl import QwenVL
from .videollama3 import VideoLLaMA3

# ── Venv reference ────────────────────────────────────────────────────────────
# .transformers-5.X-venv        → transformers 5.x    → setup.ps1
# .transformers-4.46-venv → transformers 4.46.x → setup_compat.ps1
# ─────────────────────────────────────────────────────────────────────────────

VENV = {
    "smolvlm":      ".transformers-5.X-venv",
    "qwen_vl":      ".transformers-5.X-venv",
    # videollama3 omitted — blocked on unreleased transformers VideoInput
    "moondream":    ".transformers-4.46-venv",
    "llava":        ".transformers-4.46-venv",
    "blip2":        ".transformers-4.46-venv",
    "instructblip": ".transformers-4.46-venv",
    "clip":         ".transformers-4.46-venv",
    "paligemma":    ".transformers-4.46-venv",   # also needs HF_TOKEN
    "idefics":      ".transformers-4.46-venv",
    "mplug_owl3":   ".transformers-4.46-venv",
}

REGISTRY = {
    "smolvlm":      SmolVLM,
    "qwen_vl":      QwenVL,
    # videollama3 omitted — blocked on unreleased transformers VideoInput
    "moondream":    Moondream,
    "llava":        LLaVA,
    "blip2":        BLIP2,
    "instructblip": InstructBLIP,
    "clip":         CLIP,
    "paligemma":    PaliGemma,
    "idefics":      IDEFICS,
    "mplug_owl3":   MplugOwl3,
}
