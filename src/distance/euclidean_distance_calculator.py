from dataclasses import dataclass
from src.distance.distance_calculator import DistanceCalculator
from torch import Tensor, sqrt
from numbers import Number


@dataclass
class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, x1: Tensor, x2: Tensor) -> list[Number]:
        squared_diff = (x1 - x2).pow(2)
        distances = sqrt(squared_diff.sum(dim=1))
        return [distance.item() for distance in distances]
