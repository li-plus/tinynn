import math
from collections import defaultdict
from typing import Dict

import tinynn
from tinynn.autograd import no_grad


class Optimizer:
    def __init__(self, params, defaults: Dict) -> None:
        param_groups = list(params)
        if not param_groups:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [dict(params=param_groups)]
        for param_group in param_groups:
            for name, default in defaults.items():
                param_group.setdefault(name, default)
        self.param_groups = param_groups
        self.state: Dict[tinynn.Tensor, Dict[str, tinynn.Tensor]] = defaultdict(dict)

    def step(self) -> None:
        raise NotImplementedError()

    def zero_grad(self) -> None:
        for param_group in self.param_groups:
            for p in param_group["params"]:
                p.grad = None


class SGD(Optimizer):
    def __init__(
        self, params, lr: float, momentum: float = 0, weight_decay: float = 0
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @no_grad()
    def step(self) -> None:
        for param_group in self.param_groups:
            for p in param_group["params"]:
                grad_p = p.grad.clone()

                wd = param_group["weight_decay"]
                if wd != 0:
                    grad_p += wd * p

                momentum = param_group["momentum"]
                if momentum != 0:
                    state_p = self.state[p]
                    mm_buf = state_p.get("momentum_buffer", tinynn.zeros_like(p))
                    mm_buf = momentum * mm_buf + grad_p
                    state_p["momentum_buffer"] = mm_buf
                    grad_p = mm_buf

                # TODO: do not access data, do inplace operation
                p.data -= param_group["lr"] * grad_p.data


class Adam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @no_grad()
    def step(self) -> None:
        for param_group in self.param_groups:
            for p in param_group["params"]:
                lr = param_group["lr"]
                wd = param_group["weight_decay"]
                beta1, beta2 = param_group["betas"]
                eps = param_group["eps"]

                grad_p = p.grad.clone()

                if wd != 0:
                    grad_p += wd * p

                state_p = self.state[p]
                step = state_p.get("step", tinynn.zeros(1)).item() + 1
                exp_avg = state_p.get("exp_avg", tinynn.zeros_like(p))
                exp_avg_sq = state_p.get("exp_avg_sq", tinynn.zeros_like(p))

                exp_avg = beta1 * exp_avg + (1 - beta1) * grad_p
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_p.square()

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                grad_p = exp_avg / (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps
                )

                state_p["step"] = tinynn.tensor(step)
                state_p["exp_avg"] = exp_avg
                state_p["exp_avg_sq"] = exp_avg_sq

                # TODO: do not access data, do inplace operation
                p.data -= step_size * grad_p.data
