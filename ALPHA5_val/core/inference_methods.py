"""
Alpha5 Inference Methods - Unified Interface
All inference methods in the Alpha5 project share a common interface.

This module now acts as a lightweight registry of methods, while the
implementations live in specialized modules (`*_inference.py`).
"""

from basic_inference import BasicInference
from tiled_inference import TiledInference
from multiscale_inference import MultiScaleInference
from tta_inference import TTAInference
from superres_inference import SuperResolutionInference
from hybrid_inference import HybridInference


# ============= METHODS REGISTRY =============
AVAILABLE_METHODS = {
    "basic": BasicInference(),
    "tiled": TiledInference(),
    "multiscale": MultiScaleInference(),
    "tta": TTAInference(),
    "superres": SuperResolutionInference(),
    "hybrid": HybridInference(),
}


def get_available_methods():
    """Return the list of available method keys."""
    return list(AVAILABLE_METHODS.keys())


def get_method(method_name):
    """Get a method instance by key name."""
    return AVAILABLE_METHODS.get(method_name)

