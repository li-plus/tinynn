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
        self.state: Dict[str, Dict] = defaultdict(dict)

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
                    state = self.state[p]
                    mm_buf = state.get("momentum_buffer", tinynn.zeros(p.shape))
                    mm_buf = momentum * mm_buf + grad_p
                    state["momentum_buffer"] = mm_buf
                    grad_p = mm_buf

                # TODO: do not access data, do inplace operation
                p.data -= param_group["lr"] * grad_p.data
