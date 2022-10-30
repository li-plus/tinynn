import threading
from contextlib import ContextDecorator
from typing import Any

import tinynn

_grad_mode = threading.local()
_grad_mode.grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_mode.grad_enabled


def set_grad_enabled(enabled: bool) -> None:
    _grad_mode.grad_enabled = enabled


class no_grad(ContextDecorator):
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, *exc):
        set_grad_enabled(self.prev)


class Function:
    def __call__(self, *inputs: Any) -> Any:
        with no_grad():
            output = self.forward(*inputs)
        # TODO: currently only support single output function
        assert isinstance(output, tinynn.Tensor)
        # assert output not in inputs
        requires_grad = any(
            x.requires_grad for x in inputs if isinstance(x, tinynn.Tensor)
        )
        if is_grad_enabled() and requires_grad:
            output.requires_grad = True
            output.parents = inputs
            output.grad_fn = self.backward
        return output

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError()

    def backward(self, *args: Any) -> Any:
        raise NotImplementedError()
