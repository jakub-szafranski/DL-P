import torch
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

    def __init__(self, params, lr, momentum=0.9, weight_decay=0, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1. Add weight decay to gradient
                if weight_decay != 0:
                    p.grad.add_(p, alpha=weight_decay)

                # 2. Calculate LARS adaptation
                param_norm = torch.norm(p)
                grad_norm = torch.norm(p.grad)
                adaptive_lr = 1.0

                if param_norm != 0 and grad_norm != 0:
                    adaptive_lr = eta * param_norm / (grad_norm + 1e-6)

                # 3. Scale gradient by adaptive_lr
                scaled_grad = p.grad * adaptive_lr

                # 4. SGD Step with Momentum
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(scaled_grad).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(scaled_grad, alpha=1)

                p.add_(buf, alpha=-lr)

        return loss
