from dataclasses import dataclass
from torch.nn.functional import cosine_similarity
from src.distance.distance_calculator import DistanceCalculator

@dataclass
class CosineDistanceCalculator(DistanceCalculator):
  def calculate_distance(self, input1, input2):
    return cosine_similarity(input1, input2).item()
