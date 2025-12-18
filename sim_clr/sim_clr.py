import torch
from torch import nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """SimCLR model with encoder and projection head."""

    def __init__(self, encoder: nn.Module, num_features: int, projection_dim: int = 128):
        """
        Initialize SimCLR model.

        Args:
            encoder (nn.Module): Backbone encoder (already without classification head).
            num_features (int): Output dimension of the encoder.
            projection_dim (int): Output dimension of the projection head.
        """
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim, bias=False),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        device = z_i.device
        batch_size = z_i.shape[0]
        n_samples = 2 * batch_size
        
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z, z.t())

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).reshape(n_samples, 1)

        mask = torch.ones((n_samples, n_samples), dtype=bool, device=device)
        mask.fill_diagonal_(False)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = False
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = False
        
        negatives = sim_matrix[mask].reshape(n_samples, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(n_samples, device=device).long()

        return self.criterion(logits, labels)