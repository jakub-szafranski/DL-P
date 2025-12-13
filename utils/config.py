from pydantic import BaseModel, Field, model_validator
from typing import Literal
import yaml


class Config(BaseModel):
    model: Literal["resnet50", "vit_b_16", "efficientnet_b5"] = Field(..., description="Model architecture to use.")
    model_saved_path: str = Field(..., description="Path to save model checkpoints")

    seed: int = Field(..., description="Random seed for reproducibility")
    image_size: int = Field(..., description="Input image size")

    pretrain_batch_size: int = Field(..., description="Batch size for pre-training")
    pretrain_epochs: int = Field(..., description="Number of pre-training epochs")
    pretrain_weight_decay: float = Field(..., description="Weight decay for optimizer")
    pretrain_temperature: float = Field(..., description="Temperature parameter for NT-Xent loss")

    lin_eval_batch_size: int = Field(..., description="Batch size for linear evaluation")
    lin_eval_momentum: float = Field(..., description="Momentum for linear evaluation optimizer")
    lin_eval_learning_rate: float = Field(..., description="Learning rate for linear evaluation")
    lin_eval_epochs: int = Field(..., description="Number of epochs for linear evaluation")
    lin_eval_subset_ratio: float = Field(..., description="Ratio of data used for linear evaluation")

    lin_eval_every: int = Field(..., description="Evaluate model performance every N epochs during pre-training")
    save_model_every: int = Field(..., description="Save model checkpoint every N epochs during pre-training")

    @model_validator(mode="after")
    def check_save_interval(self):
        if self.save_model_every % self.lin_eval_every != 0:
            raise ValueError("save_model_every must be a multiple of lin_eval_every")
        return self


with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

conf = Config(**config_dict)
