from typing import Literal
import torch.nn as nn
import torchvision


def get_encoder(name: Literal["resnet50", "vit_b_16", "efficientnet_b5"]) -> nn.Module:
    """
    Returns a pre-defined encoder model based on the specified name.

    Args:
        name (Literal): Name of the encoder model to retrieve.

    Returns:
        nn.Module: The requested encoder model without the classification head.

    Raises:
        KeyError: If the specified name is not a valid encoder model.
    """
    if name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Identity()

    elif name == "vit_b_16":
        model = torchvision.models.vit_b_16(weights=None)
        model.heads.head = nn.Identity()

    elif name == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Identity()

    else:
        raise KeyError(f"{name} is not a valid encoder version. Choose from: resnet50, vit_b_16, efficientnet_b5")

    return model
