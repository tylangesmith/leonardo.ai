from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image
from torch import Tensor


@dataclass
class Input:
    """
    Represents the input for zero-shot image classification, consisting of an image and its corresponding caption.

    Attributes:
        image (Image): An image object, typically loaded using PIL or a similar library.
        caption (str): A textual description or caption associated with the image.
    """

    image: Image
    caption: str


@dataclass
class Prediction:
    """
    Represents the output of a zero-shot image classification model, including embeddings for both image and text.

    Attributes:
        image_embedding (Tensor): A tensor representing the embedding of the input image.
        text_embedding (Tensor): A tensor representing the embedding of the input text or caption.
    """

    image_embedding: Tensor
    text_embedding: Tensor


@dataclass
class ZeroShotImageClassificationModel(ABC):
    """
    An abstract base class for zero-shot image classification models.

    This class defines an interface for models that can generate predictions (embeddings) for a given set of inputs,
    each comprising an image and a textual description. Implementations of this class must define the `predict`
    method, detailing how inputs are transformed into embeddings.

    Methods:
        predict(inputs: list[Input]) -> Prediction: Abstract method to generate predictions from inputs.
    """

    @abstractmethod
    def predict(self, inputs: list[Input]) -> Prediction:
        """
        Generates predictions for a given list of inputs, where each input includes an image and its caption.

        Args:
            inputs (list[Input]): A list of `Input` instances, each containing an image and a caption.

        Returns:
            Prediction: A `Prediction` instance containing the embeddings for the images and text in the inputs.
        """
        pass
