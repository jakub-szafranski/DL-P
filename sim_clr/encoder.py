from typing import Literal
import torch.nn as nn
import torchvision


def get_encoder(name: Literal["resnet50", "vit_b_16", "efficientnet_b5"]) -> tuple[nn.Module, int]:
    """
    Returns a pre-defined encoder model and its feature dimension.

    Args:
        name (Literal): Name of the encoder model to retrieve.

    Returns:
        tuple[nn.Module, int]: The encoder model and its output feature dimension.

    Raises:
        KeyError: If the specified name is not a valid encoder model.
    """
    if name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Identity()

    elif name == "vit_b_16":
        model = torchvision.models.vit_b_16(weights=None)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()

    elif name == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()

    else:
        raise KeyError(f"{name} is not a valid encoder version. Choose from: resnet50, vit_b_16, efficientnet_b5")

    return model, num_features
