"""
SoftMatch training utilities derived from the TorchSSL implementation: https://github.com/TorchSSL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SoftMatchTrainer:
    """SoftMatch trainer for semi-supervised learning with SimCLR."""

    def __init__(
        self,
        num_classes: int,
        dist_align: bool = True,
        hard_label: bool = True,
        ema_p: float = 0.999,
        device: torch.device = None,
    ):
        """
        Args:
            num_classes (int): Number of classification classes.
            dist_align (bool): Whether to apply distribution alignment.
            hard_label (bool): Use hard pseudo labels if True.
            ema_p (float): EMA decay for probability tracking.
            device (torch.device): Device for tensors.
        """
        self.num_classes = num_classes
        self.dist_align = dist_align
        self.hard_label = hard_label
        self.ema_p = ema_p
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize probability tracking tensors
        self.lb_prob_t = torch.ones(num_classes, device=self.device) / num_classes
        self.ulb_prob_t = torch.ones(num_classes, device=self.device) / num_classes
        self.prob_max_mu_t = 1.0 / num_classes
        self.prob_max_var_t = 1.0

    @torch.no_grad()
    def update_prob_t(self, lb_probs: torch.Tensor, ulb_probs: torch.Tensor) -> None:
        """
        Update probability tracking statistics using EMA.

        Args:
            lb_probs (torch.Tensor): Labeled data class probabilities.
            ulb_probs (torch.Tensor): Unlabeled data class probabilities.
        """
        ulb_prob_t = ulb_probs.mean(0)
        lb_prob_t = lb_probs.mean(0)

        max_probs, _ = ulb_probs.max(dim=-1)
        max_prob_sum = max_probs.sum()
        max_prob_sumsq = (max_probs**2).sum()
        max_prob_count = torch.tensor(max_probs.numel(), device=max_probs.device, dtype=max_prob_sum.dtype)

        if dist.is_initialized():
            world_size = dist.get_world_size()

            dist.all_reduce(ulb_prob_t, op=dist.ReduceOp.SUM)
            ulb_prob_t /= world_size
            dist.all_reduce(lb_prob_t, op=dist.ReduceOp.SUM)
            lb_prob_t /= world_size

            dist.all_reduce(max_prob_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_prob_sumsq, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_prob_count, op=dist.ReduceOp.SUM)

        prob_max_mu_t = max_prob_sum / max_prob_count
        prob_max_var_t = (max_prob_sumsq / max_prob_count) - (prob_max_mu_t**2)
        prob_max_var_t = torch.clamp(prob_max_var_t, min=1e-6)

        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()

    @torch.no_grad()
    def _calculate_mask(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate sample weights based on prediction confidence.

        Args:
            probs (torch.Tensor): Class probabilities [B, num_classes].

        Returns:
            torch.Tensor: Mask weights.
        """
        max_probs, _ = probs.max(dim=-1)
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4)))
        return mask.detach()

    @torch.no_grad()
    def _distribution_alignment(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Apply distribution alignment to balance class predictions.

        Args:
            probs (torch.Tensor): Class probabilities [B, num_classes].

        Returns:
            torch.Tensor: Aligned probabilities.
        """
        probs = probs * self.lb_prob_t / self.ulb_prob_t
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()

    def compute_unsupervised_loss(
        self,
        logits_weak: torch.Tensor,
        logits_strong: torch.Tensor,
        probs_weak: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute unsupervised consistency loss between weak and strong augmentations.

        Args:
            logits_weak (torch.Tensor): Logits from weak augmentation [B, num_classes].
            logits_strong (torch.Tensor): Logits from strong augmentation [B, num_classes].
            probs_weak (torch.Tensor): Optional precomputed weak probs [B, num_classes].

        Returns:
            torch.Tensor: Unsupervised consistency loss.
        """
        if probs_weak is None:
            probs_weak = torch.softmax(logits_weak.detach(), dim=-1)

        if self.dist_align:
            probs_aligned = self._distribution_alignment(probs_weak)
        else:
            probs_aligned = probs_weak

        mask = self._calculate_mask(probs_aligned)

        loss, _ = consistency_loss(
            logits_strong,
            logits_weak,
            mask,
            name="ce",
            use_hard_labels=self.hard_label,
        )
        return loss

    def compute_supervised_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised cross-entropy loss for labeled data.

        Args:
            logits (torch.Tensor): Model logits [B, num_classes].
            targets (torch.Tensor): Ground truth labels [B].

        Returns:
            torch.Tensor: Supervised loss scalar.
        """
        return ce_loss(logits, targets, use_hard_labels=True, reduction="mean")


class ModelEMA:
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model (nn.Module): Model to track.
            decay (float): EMA decay factor.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._registered = False

    def register(self) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._registered = True

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights with EMA."""
        if not self._registered:
            self.register()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow"
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Apply shadow weights to model (backup current weights)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
    

def consistency_loss(logits_s, logits_w, mask, name='ce', T=0.5, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        _, max_idx = torch.max(pseudo_label, dim=-1)
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
        return masked_loss.mean(), mask

    else:
        assert Exception('Not Implemented consistency_loss')