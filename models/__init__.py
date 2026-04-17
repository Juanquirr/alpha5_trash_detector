from .smolvlm import SmolVLM
from .moondream import Moondream

REGISTRY = {
    "smolvlm": SmolVLM,
    "moondream": Moondream,
}
