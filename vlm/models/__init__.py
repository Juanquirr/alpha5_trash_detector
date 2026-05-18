from .smolvlm import SmolVLM
from .smolvlm_500m import SmolVLM500M
from .qwen_vl import QwenVL
from .qwen2b import QwenVL2B
from .llava_ov import LlavaOV
from .llava import LLaVA
from .moondream import Moondream
from .clip import CLIP
from .internvl2 import InternVL2

# ── Venv reference ────────────────────────────────────────────────────────────
# .transformers-5.X-venv  → transformers 5.x    → setup.ps1
# .transformers-4.46-venv → transformers 4.46.x → setup_compat.ps1
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: llava_ov (lmms-lab/llava-onevision-qwen2-0.5b-ov) is incompatible
# with both available venvs:
#   5.x : key structure refactored — ALL checkpoint weights MISSING/UNEXPECTED
#   4.46: SiGLIP strict check fails (embed_dim=1152, num_heads=14, 1152%14≠0)
# Kept in REGISTRY for reference but excluded from pope_run / pope_finetune.
# ─────────────────────────────────────────────────────────────────────────────

VENV = {
    "smolvlm":      ".transformers-5.X-venv",
    "smolvlm_500m": ".transformers-5.X-venv",
    "qwen_vl":      ".transformers-5.X-venv",
    "qwen_2b":      ".transformers-5.X-venv",
    "llava_ov":     ".transformers-4.46-venv",   # kept but non-functional
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
    "llava_ov":     LlavaOV,
    "llava":        LLaVA,
    "moondream":    Moondream,
    "clip":         CLIP,
    "internvl2":    InternVL2,
}
