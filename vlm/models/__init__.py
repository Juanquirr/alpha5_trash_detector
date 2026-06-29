from .smolvlm import SmolVLM
from .smolvlm_500m import SmolVLM500M
from .qwen_vl import QwenVL
from .qwen2b import QwenVL2B
from .llava import LLaVA

# ── Venv reference ────────────────────────────────────────────────────────────
# .transformers-5.X-venv  → transformers 5.x    → setup_5.x.ps1
# .transformers-4.46-venv → transformers 4.46.x → setup_4.46.ps1
# ─────────────────────────────────────────────────────────────────────────────

VENV = {
    "smolvlm":      ".transformers-5.X-venv",
    "smolvlm_500m": ".transformers-5.X-venv",
    "qwen25_vl":    ".transformers-5.X-venv",
    "qwen3_vl":     ".transformers-5.X-venv",
    "llava":        ".transformers-4.46-venv",
}

REGISTRY = {
    "smolvlm":      SmolVLM,
    "smolvlm_500m": SmolVLM500M,
    "qwen25_vl":    QwenVL,
    "qwen3_vl":     QwenVL2B,
    "llava":        LLaVA,
}
