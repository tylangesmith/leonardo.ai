from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number

@dataclass
class DistanceCalculator(ABC):
  @abstractmethod
  def calculate_distance(self, input1, input2) -> Number:
    pass
