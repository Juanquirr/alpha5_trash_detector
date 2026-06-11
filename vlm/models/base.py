import torch
from abc import ABC, abstractmethod


DEFAULT_CLASSES = [
    "container",
    "plastic",
    "metal",
    "polystyrene",
    "plastic_fragment",
    "trash_pile",
    "trash",
]


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

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
