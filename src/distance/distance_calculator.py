from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from torch import Tensor


@dataclass
class DistanceCalculator(ABC):
    """
    An abstract base class for distance calculations.

    This class serves as an interface for implementing various distance calculation
    strategies between tensors. Subclasses must implement the `calculate_distance`
    method, which calculates and returns the distance between two tensors.

    Methods:
        calculate_distance(x1: Tensor, x2: Tensor) -> list[Number]: Abstract method to calculate the distance.
    """

    @abstractmethod
    def calculate_distance(self, x1: Tensor, x2: Tensor) -> list[Number]:
        """
        Calculates the distance between two tensors.

        This is an abstract method that must be implemented by subclasses. The
        method should return a list of numerical values representing the calculated
        distances between elements of the input tensors.

        Args:
            x1 (Tensor): The first tensor.
            x2 (Tensor): The second tensor, to be compared with the first.

        Returns:
            list[Number]: A list of distances between elements of the input tensors.
        """
        pass
