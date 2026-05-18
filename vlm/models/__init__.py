from .smolvlm import SmolVLM
from .smolvlm_500m import SmolVLM500M
from .qwen_vl import QwenVL
from .qwen2b import QwenVL2B
from .llava import LLaVA
from .moondream import Moondream
from .clip import CLIP
from .internvl2 import InternVL2

# ── Venv reference ────────────────────────────────────────────────────────────
# .transformers-5.X-venv  → transformers 5.x    → setup.ps1
# .transformers-4.46-venv → transformers 4.46.x → setup_compat.ps1
# ─────────────────────────────────────────────────────────────────────────────

VENV = {
    "smolvlm":      ".transformers-5.X-venv",
    "smolvlm_500m": ".transformers-5.X-venv",
    "qwen_vl":      ".transformers-5.X-venv",
    "qwen_2b":      ".transformers-5.X-venv",
    "llava":        ".transformers-4.46-venv",
    "moondream":    ".transformers-4.46-venv",
    "clip":         ".transformers-4.46-venv",
    "internvl2":    ".transformers-4.46-venv",
}

REGISTRY = {
    "smolvlm":      SmolVLM,
    "smolvlm_500m": SmolVLM500M,
    "qwen_vl":      QwenVL,
    "qwen_2b":      QwenVL2B,
    "llava":        LLaVA,
    "moondream":    Moondream,
    "clip":         CLIP,
    "internvl2":    InternVL2,
}
