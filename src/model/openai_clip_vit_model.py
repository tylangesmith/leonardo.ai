from dataclasses import dataclass, field
from typing import TypeAlias, Literal
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput
from src.model.zero_shot_image_classification_model import (
    ZeroShotImageClassificationModel,
    Prediction,
    Input,
)


_default_model = "openai/clip-vit-base-patch16"
_supported_model_names: TypeAlias = Literal[
    "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"
]


@dataclass
class OpenaiClipVitModel(ZeroShotImageClassificationModel):
    """
    An implementation of ZeroShotImageClassificationModel using OpenAI's CLIP models.

    Attributes:
        model_name (_supported_model_names): The name of the pre-trained CLIP model to be used.
        processor (CLIPProcessor): The processor for preparing inputs for the CLIP model.
        model (CLIPModel): The CLIP model.

    Methods:
        predict(inputs: list[Input]) -> Prediction: Processes inputs and predicts embeddings.
    """

    model_name: _supported_model_names = field(default=_default_model)
    processor: CLIPProcessor = field(init=False)
    model: CLIPModel = field(init=False)

    def __post_init__(self):
        """
        Initializes the processor and model based on the specified or default model name.
        """
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

    def predict(self, inputs: list[Input]) -> Prediction:
        """
        Predicts image and text embeddings for a given list of inputs using the CLIP model.

        Each input is processed to extract image and text embeddings, which are then returned
        as part of a Prediction object.

        Args:
            inputs (list[Input]): A list of Input objects, each containing an image and a caption.

        Returns:
            Prediction: An object containing the embeddings for images and text derived from the inputs.
        """
        images = [input.image for input in inputs]
        captions = [input.caption for input in inputs]

        processed_inputs = self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs: CLIPOutput = self.model(**processed_inputs)

        return Prediction(
            image_embedding=outputs.image_embeds, text_embedding=outputs.text_embeds
        )
