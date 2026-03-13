import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DatasetSaver(ABC):
    def __init__(self):
        self.train_images = []
        self.train_labels = {}
        self.validation_images = []
        self.validation_labels = {}

    def add_training(self, img, bounding_box: Tuple[float, float, float, float]):
        md5 = hashlib.md5(img.tobytes()).hexdigest()
        if not md5 in self.train_labels:
            self.train_labels[md5] = []
            self.train_images.append(img)
        self.train_labels[md5].append(bounding_box)


    def add_validation(self, img, bounding_box: Tuple[float, float, float, float]):
        md5 = hashlib.md5(img.tobytes()).hexdigest()
        if not md5 in self.train_labels:
            self.validation_labels[md5] = []
            self.validation_images.append(img)
        self.validation_labels[md5].append(bounding_box)

    def get_train_labels_from_image(self, img) -> List[Tuple[float, float, float, float]]:
        return self.train_labels[hashlib.md5(img.tobytes()).hexdigest()]

    def get_val_labels_from_image(self, img) -> List[Tuple[float, float, float, float]]:
        return self.validation_labels[hashlib.md5(img.tobytes()).hexdigest()]


    @abstractmethod
    def save(self, path: str):
        ...