from dataclasses import dataclass
from numbers import Number
from src.model.zero_shot_image_classification_model import (
    ZeroShotImageClassificationModel,
    Input,
)
from src.distance.distance_calculator import DistanceCalculator


@dataclass
class ImageTextSimilarityCalculator:
    model: ZeroShotImageClassificationModel
    distance_calculator: DistanceCalculator

    def calculate(self, inputs: list[Input]) -> list[Number]:
        # Make our predictions, in this case because it's a zero-shot image
        # classification model we'll get the image and text embeddings
        prediction = self.model.predict(inputs)

        # Calculate the distance between the image and text embeddings
        return self.distance_calculator.calculate_distance(
            prediction.image_embedding, prediction.text_embedding
        )
