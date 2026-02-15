from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import yaml
import tyro


class SimCLRConfig(BaseModel):
    """Configuration for SimCLR training."""

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

    # Fine-tuning options
    ft_subset_ratios: list[float] = Field(..., description="List of data ratios used for fine-tuning evaluation")
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
    eval_model_paths: list[str] = Field(..., description="List of model filepaths for evaluation")

    @model_validator(mode="after")
    def check_save_interval(self):
        if self.save_model_every % self.eval_every != 0:
            raise ValueError("save_model_every must be a multiple of eval_every")
        return self


class SoftCLRConfig(BaseModel):
    """Configuration for SoftMatch + SimCLR training."""

    model_saved_path: str = Field(..., description="Path to save model checkpoints")
    models: list[str] = Field(..., description="List of encoder backbones to use")
    seed: int = Field(..., description="Random seed for reproducibility")
    image_size: int = Field(..., description="Input image size")

    # Pre-training options
    pretrain_batch_size: int = Field(..., description="Batch size for pre-training")
    pretrain_learning_rate: float = Field(..., description="Learning rate for pre-training")
    pretrain_epochs: int = Field(..., description="Number of pre-training epochs")
    pretrain_weight_decay: float = Field(..., description="Weight decay for optimizer")
    pretrain_momentum: float = Field(..., description="Momentum for optimizer")
    pretrain_temperature: float = Field(..., description="Temperature parameter for NT-Xent loss")

    # Evaluation and saving intervals
    eval_every: int = Field(..., description="Evaluate model performance every N epochs during pre-training")
    save_model_every: int = Field(..., description="Save model checkpoint every N epochs during pre-training")

    # SoftMatch options
    softmatch_sup_weight: float = Field(..., description="Weight for supervised loss")
    softmatch_unsup_weight: float = Field(..., description="Weight for unsupervised consistency loss")
    simclr_weight: float = Field(..., description="Weight for SimCLR loss")
    softmatch_dist_align: bool = Field(..., description="Apply distribution alignment")
    softmatch_hard_label: bool = Field(..., description="Use hard pseudo labels")
    softmatch_ema_p: float = Field(..., description="EMA decay for probability tracking")
    softmatch_model_ema: float = Field(..., description="EMA decay for model weights")
    soft_weights_epoch: int = Field(..., description="Epoch to start applying soft weights to unsupervised loss")
    softmatch_loss_weight: float = Field(..., description="Multiplier for softmatch loss in total loss")
    use_decaying_loss_weight: bool = Field(..., description="Whether to use decaying loss weight for unsupervised loss")
    decaying_weight_min: float = Field(..., description="Minimum weight for decaying loss weight")
    decaying_weight_max: float = Field(..., description="Maximum weight for decaying loss weight")

    @model_validator(mode="after")
    def check_save_interval(self):
        if self.save_model_every % self.eval_every != 0:
            raise ValueError("save_model_every must be a multiple of eval_every")
        return self


# Load SimCLR config
simclr_path = Path(__file__).parent.parent / "config_simclr.yaml"
with open(simclr_path, "r") as f:
    simclr_config_dict = yaml.safe_load(f)
conf_simclr = SimCLRConfig(**simclr_config_dict)

# Load SimCLR + SoftMatch config
softmatch_path = Path(__file__).parent.parent / "config_softclr.yaml"
with open(softmatch_path, "r") as f:
    softmatch_config_dict = yaml.safe_load(f)
conf_softclr = SoftCLRConfig(**softmatch_config_dict)


def parse_simclr_cli() -> SimCLRConfig:
    """Parse SimCLR config from YAML defaults with CLI overrides."""
    return tyro.cli(SimCLRConfig, default=conf_simclr)


def parse_softclr_cli() -> SoftCLRConfig:
    """Parse SoftCLR config from YAML defaults with CLI overrides."""
    return tyro.cli(SoftCLRConfig, default=conf_softclr)
