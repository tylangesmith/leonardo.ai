from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class Input:
  image: Image
  caption: str


@dataclass
class Prediction:
  image_embedding: any
  text_embedding: any


@dataclass
class ZeroShotImageClassificationModel(ABC):
  @abstractmethod
  def predict(self, inputs: list[Input]) -> list[Prediction]:
    pass
