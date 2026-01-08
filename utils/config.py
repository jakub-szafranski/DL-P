from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import yaml


class Config(BaseModel):
    model_saved_path: str = Field(..., description="Path to save model checkpoints")

    models: list[str] = Field(..., description="List of encoder backbones to use")

    seed: int = Field(..., description="Random seed for reproducibility")
    image_size: int = Field(..., description="Input image size")

    # Pre-training options
    pretrain_batch_size: int = Field(..., description="Batch size for pre-training")
    pretrain_learning_rate: float = Field(..., description="Learning rate for pre-training")
    pretrain_epochs: int = Field(..., description="Number of pre-training epochs")
    pretrain_weight_decay: float = Field(..., description="Weight decay for optimizer")
    pretrain_temperature: float = Field(..., description="Temperature parameter for NT-Xent loss")

    # SoftMatch options
    softmatch_subset_ratio: float = Field(..., description="Ratio of labeled data used for supervised loss")
    softmatch_sup_weight: float = Field(..., description="Weight for supervised loss")
    softmatch_unsup_weight: float = Field(..., description="Weight for unsupervised consistency loss")
    softmatch_dist_align: bool = Field(..., description="Apply distribution alignment")
    softmatch_hard_label: bool = Field(..., description="Use hard pseudo labels")
    softmatch_ema_p: float = Field(..., description="EMA decay for probability tracking")
    softmatch_model_ema: float = Field(..., description="EMA decay for model weights")

    # Fine-tuning options
    ft_subset_ratio: float = Field(..., description="Ratio of data used for fine-tuning")

    ft_frozen_batch_size: int = Field(..., description="Batch size for frozen encoder stage")
    ft_frozen_learning_rate: float = Field(..., description="Learning rate for frozen encoder stage")
    ft_frozen_epochs: int = Field(..., description="Number of epochs for frozen encoder stage")
    ft_frozen_momentum: float = Field(..., description="Momentum for frozen encoder stage")

    ft_full_batch_size: int = Field(..., description="Batch size for full fine-tuning")
    ft_full_learning_rate: float = Field(..., description="Learning rate for full fine-tuning")
    ft_full_epochs: int = Field(..., description="Number of epochs for full fine-tuning")
    ft_full_momentum: float = Field(..., description="Momentum for full fine-tuning")

    # Evaluation and saving intervals
    eval_every: int = Field(..., description="Evaluate model performance every N epochs during pre-training")
    save_model_every: int = Field(..., description="Save model checkpoint every N epochs during pre-training")

    @model_validator(mode="after")
    def check_save_interval(self):
        if self.save_model_every % self.eval_every != 0:
            raise ValueError("save_model_every must be a multiple of eval_every")
        return self


path = Path(__file__).parent.parent / "config.yaml"
with open(path, "r") as f:
    config_dict = yaml.safe_load(f)

conf = Config(**config_dict)
