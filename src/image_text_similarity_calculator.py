from dataclasses import dataclass
from numbers import Number
from src.model.zero_shot_image_classification_model import (
    ZeroShotImageClassificationModel,
    Input,
)
from src.distance.distance_calculator import DistanceCalculator


@dataclass
class ImageTextSimilarityCalculator:
    """
    A class for determining the similarity between images and text using a
    zero-shot image classification model and a distance calculator.

    Attributes:
        model (ZeroShotImageClassificationModel): The zero-shot image classification model used to get embeddings.
        distance_calculator (DistanceCalculator): The calculator used to compute the distance between embeddings.

    Methods:
        calculate(inputs: list[Input]) -> list[Number]: Calculates the similarity scores between given images and text.
    """

    model: ZeroShotImageClassificationModel
    distance_calculator: DistanceCalculator

    def calculate(self, inputs: list[Input]) -> list[Number]:
        """
        Calculates the similarity scores between images and text by first obtaining their embeddings
        from a zero-shot image classification model, then computing the distances between these embeddings.

        Args:
            inputs (list[Input]): A list of inputs where each input contains both the image and the text to be compared.

        Returns:
            list[Number]: A list of numerical similarity scores corresponding to each input pair of image and text,
                          where a lower score indicates greater similarity.
        """
        prediction = self.model.predict(inputs)

        return self.distance_calculator.calculate_distance(
            prediction.image_embedding, prediction.text_embedding
        )
