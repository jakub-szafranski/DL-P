import torch.nn as nn

class SoftCLR(nn.Module):
    """SoftCLR model with encoder, projection and classification head."""

    def __init__(self, encoder: nn.Module, num_features: int, projection_dim: int = 128, num_classes: int = 1000):
        """
        Args:
            encoder (nn.Module): Backbone encoder (already without classification head).
            num_features (int): Output dimension of the encoder.
            projection_dim (int): Output dimension of the projection head.
            num_classes (int): Number of classes for classification head.
        """
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim, bias=False),
        )
        self.classification_head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection_head(features)
        classification = self.classification_head(features)
        return projection, classification