import torch
from collections.abc import Callable, Iterable
from typing import Any
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Base learning rate.
        momentum (float, optional): Momentum factor (default: 0.9).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        eta (float, optional): LARS coefficient (default: 0.001).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the LARS optimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): Parameters to optimize (or parameter groups).
            lr (float): Base learning rate.
            momentum (float): Momentum factor.
            weight_decay (float): Weight decay factor.
            trust_coefficient (float): Trust coefficient used for layer-wise adaptation.
            eps (float): Small constant for numerical stability.
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform a single optimization step.

        Args:
            closure (Callable[[], torch.Tensor] | None): Optional closure that reevaluates the model.

        Returns:
            torch.Tensor | None: The loss, if provided by the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            trust_coefficient = group["trust_coefficient"]
            eps = group["eps"]
            layer_adaptation = group.get("layer_adaptation", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                is_1d_param = p.ndim == 1

                # 1) Weight decay (skip 1D params like bias/BN)
                if weight_decay != 0 and not is_1d_param:
                    grad.add_(p, alpha=weight_decay)

                # 2) LARS layer-wise adaptation (skip 1D params)
                adaptive_lr = 1.0
                if layer_adaptation and not is_1d_param:
                    param_norm = torch.norm(p)
                    grad_norm = torch.norm(grad)
                    if param_norm != 0 and grad_norm != 0:
                        adaptive_lr = trust_coefficient * param_norm / (grad_norm + eps)

                # 3) Scale gradient by adaptive_lr
                scaled_grad = grad * adaptive_lr

                # 4) SGD step with momentum
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(scaled_grad).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(scaled_grad, alpha=1)

                p.add_(buf, alpha=-lr)

        return loss
