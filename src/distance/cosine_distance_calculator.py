from dataclasses import dataclass
from torch.nn.functional import cosine_similarity
from torch import Tensor
from numbers import Number
from src.distance.distance_calculator import DistanceCalculator


@dataclass
class CosineDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, x1: Tensor, x2: Tensor) -> list[Number]:
        similarities = cosine_similarity(x1, x2)
        return [similarity.item() for similarity in similarities]
