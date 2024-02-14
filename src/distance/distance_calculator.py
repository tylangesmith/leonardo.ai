from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from torch import Tensor


@dataclass
class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, x1: Tensor, x2: Tensor) -> list[Number]:
        pass
