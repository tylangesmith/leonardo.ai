from dataclasses import dataclass
from src.distance.distance_calculator import DistanceCalculator


@dataclass
class EuclideanDistanceCalculator(DistanceCalculator):
  def calculate_distance(self, input1, input2):
    pass
