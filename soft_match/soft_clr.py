import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn


class SoftCLR(nn.Module):
    """SoftCLR model with encoder, projection and classification head."""

    def __init__(self, encoder: nn.Module, num_features: int, projection_dim: int = 128, num_classes: int = 10):
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


class SoftNTXentLoss(nn.Module):
    """
    Soft NT-Xent loss with pseudo-label weighting for false negative mitigation.
    
    Standard NT-Xent: ℓ_i,j = -sim(z_i,z_j)/τ + log(Σ_k 𝟙[k≠i] · exp(sim(z_i,z_k)/τ))
    
    Soft NT-Xent:     ℓ_i,j = -sim(z_i,z_j)/τ + log(Σ_k w_k · 𝟙[k≠i] · exp(sim(z_i,z_k)/τ))
    
    where w_k ∈ [0,1] reduces repulsion for samples predicted to be same class.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        probs_i: torch.Tensor,
        probs_j: torch.Tensor,
        prob_max_mu: float,
        prob_max_var: float,
        use_weights: bool,
    ) -> torch.Tensor:
        """
        Compute Soft NT-Xent loss with pseudo-label weighting.

        Args:
            z_i (torch.Tensor): Embeddings from first augmentation.
            z_j (torch.Tensor): Embeddings from second augmentation.
            probs_i (torch.Tensor): Class probabilities for z_i.
            probs_j (torch.Tensor): Class probabilities for z_j.
            prob_max_mu (float): EMA mean of max probabilities from SoftMatchTrainer.
            prob_max_var (float): EMA variance of max probabilities from SoftMatchTrainer.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        device = z_i.device

        # Gather embeddings and probs from all GPUs
        if dist.is_initialized():
            z_i_all = torch.cat(dist_nn.functional.all_gather(z_i), dim=0)
            z_j_all = torch.cat(dist_nn.functional.all_gather(z_j), dim=0)
            probs_i_all = torch.cat(dist_nn.functional.all_gather(probs_i), dim=0)
            probs_j_all = torch.cat(dist_nn.functional.all_gather(probs_j), dim=0)
        else:
            z_i_all, z_j_all = z_i, z_j
            probs_i_all, probs_j_all = probs_i, probs_j

        batch_size = z_i_all.shape[0]
        n_samples = 2 * batch_size

        z = F.normalize(torch.cat([z_i_all, z_j_all], dim=0), dim=1)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        if use_weights:
            weights = self._compute_soft_weights(probs_i_all, probs_j_all, prob_max_mu, prob_max_var)
        else:
            weights = torch.ones_like(sim_matrix)

        pos_mask = torch.zeros((n_samples, n_samples), dtype=torch.bool, device=device)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        self_mask = torch.eye(n_samples, dtype=torch.bool, device=device)
        neg_mask = ~(self_mask | pos_mask)

        pos_sim = sim_matrix[pos_mask].view(n_samples)

        log_weights = torch.where(
            weights > 0, 
            torch.log(weights), 
            torch.tensor(float('-inf'), device=device)
        )
        weighted_sim = sim_matrix + log_weights
        weighted_sim[~neg_mask] = float('-inf')

        all_terms = torch.cat([pos_sim.unsqueeze(1), weighted_sim], dim=1)
        log_denom = torch.logsumexp(all_terms, dim=1)

        loss = log_denom - pos_sim

        return loss.mean()

    def _compute_soft_weights(
        self,
        probs_i: torch.Tensor,
        probs_j: torch.Tensor,
        prob_max_mu: float,
        prob_max_var: float,
    ) -> torch.Tensor:
        """
        Compute soft weights using SoftMatch Gaussian formula.
        
        For each pair (a, b):
          w_a,b = 1 - (mask_a · mask_b · 𝟙[ŷ_a == ŷ_b])
        
        where mask uses truncated Gaussian (SoftMatch):
          mask = exp(-((clamp(conf - μ, max=0))² / (var/2)))

        Args:
            probs_i (torch.Tensor): Class probabilities for first view.
            probs_j (torch.Tensor): Class probabilities for second view.
            prob_max_mu (float): EMA mean of max probabilities.
            prob_max_var (float): EMA variance of max probabilities.

        Returns:
            torch.Tensor: Weight matrix [2N, 2N] in range [0, 1].
        """
        conf_i, pred_i = probs_i.max(dim=1)
        conf_j, pred_j = probs_j.max(dim=1)

        conf_all = torch.cat([conf_i, conf_j], dim=0)
        pred_all = torch.cat([pred_i, pred_j], dim=0)

        mask_all = torch.exp(-((torch.clamp(conf_all - prob_max_mu, max=0.0) ** 2) / (2 * prob_max_var / 4)))

        mask_product = mask_all.unsqueeze(1) * mask_all.unsqueeze(0)
        same_class = (pred_all.unsqueeze(1) == pred_all.unsqueeze(0)).float()

        return (1.0 - mask_product * same_class).clamp(min=1e-8, max=1.0)