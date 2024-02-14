from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image
from torch import Tensor


@dataclass
class Input:
    image: Image
    caption: str


@dataclass
class Prediction:
    image_embedding: Tensor
    text_embedding: Tensor


@dataclass
class ZeroShotImageClassificationModel(ABC):
    @abstractmethod
    def predict(self, inputs: list[Input]) -> Prediction:
        pass
