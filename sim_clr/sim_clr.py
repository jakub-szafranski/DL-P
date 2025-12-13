import torch
from torch import nn


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim, bias=False),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cuda:0"):
        super().__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

        self.device = device

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positive_samples = torch.cat([sim_ij, sim_ji], dim=0).reshape(2 * batch_size, 1)

        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).to(self.device)
        negative_samples = sim_matrix[mask].reshape(2 * batch_size, -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size).to(self.device).long()

        return self.criterion(logits, labels)
