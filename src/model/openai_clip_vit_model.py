from dataclasses import dataclass, field
from typing import Literal
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput
from src.model.zero_shot_image_classification_model import (
    ZeroShotImageClassificationModel,
    Prediction,
    Input,
)


@dataclass
class OpenaiClipVitModel(ZeroShotImageClassificationModel):
    model_name: Literal[
        "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"
    ] = field(default="openai/clip-vit-base-patch16")
    processor: CLIPProcessor = field(init=False)
    model: CLIPModel = field(init=False)

    def __post_init__(self):
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

    def predict(self, inputs: list[Input]) -> Prediction:
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
