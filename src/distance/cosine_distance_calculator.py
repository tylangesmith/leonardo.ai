from dataclasses import dataclass
from torch.nn.functional import cosine_similarity
from torch import Tensor
from numbers import Number
from src.distance.distance_calculator import DistanceCalculator


@dataclass
class CosineDistanceCalculator(DistanceCalculator):
    """
    Implements the DistanceCalculator for computing the cosine similarity between tensors.

    Methods:
        calculate_distance(x1: Tensor, x2: Tensor) -> list[Number]: Calculates the cosine similarity.
    """

    def calculate_distance(self, x1: Tensor, x2: Tensor) -> list[Number]:
        """
        Calculates the cosine similarity between two tensors.

        Computes the cosine similarity for each pair of rows in the input tensors `x1` and `x2`.
        The resulting similarity scores are returned as a list, with higher values indicating
        greater similarity.

        Args:
            x1 (Tensor): The first tensor, each row representing a vector.
            x2 (Tensor): The second tensor, each row representing a vector to be compared with the first.

        Returns:
            list[Number]: A list containing the cosine similarity scores for each pair of corresponding rows from `x1` and `x2`.

        Note: cosine_similarity returns values in [-1, 1], with 1 meaning identical direction.
        """
        similarities = cosine_similarity(x1, x2)
        return [similarity.item() for similarity in similarities]
