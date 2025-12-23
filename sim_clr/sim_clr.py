import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
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
    """NT-Xent loss for contrastive learning with distributed support."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss with all_gather for distributed training.

        Args:
            z_i (torch.Tensor): Embeddings from first augmentation view.
            z_j (torch.Tensor): Embeddings from second augmentation view.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        device = z_i.device

        # Gather embeddings from all GPUs for full negative set
        if dist.is_initialized():
            z_i_gathered = dist_nn.functional.all_gather(z_i)
            z_j_gathered = dist_nn.functional.all_gather(z_j)
            z_i_all = torch.cat(z_i_gathered, dim=0)
            z_j_all = torch.cat(z_j_gathered, dim=0)
        else:
            z_i_all = z_i
            z_j_all = z_j

        batch_size = z_i_all.shape[0]
        n_samples = 2 * batch_size

        z = torch.cat([z_i_all, z_j_all], dim=0)
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
