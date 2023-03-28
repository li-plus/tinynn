import threading
from contextlib import ContextDecorator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import tinynn

_TensorOrTensors = Union["tinynn.Tensor", Sequence["tinynn.Tensor"]]


def _make_tensor_tuple(tensor_or_tensors: _TensorOrTensors) -> Tuple["tinynn.Tensor"]:
    tensors = tensor_or_tensors
    if not isinstance(tensors, Sequence):
        tensors = (tensors,)
    return tensors


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


class FunctionNode:
    def __init__(self, fn: Callable, inputs: Any, outputs: Any) -> None:
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class FunctionCtx:
    def __init__(self) -> None:
        self.saved_tensors: Tuple[tinynn.Tensor, ...] = None

    def save_for_backward(self, *args: Any) -> None:
        self.saved_tensors = _make_tensor_tuple(args)


class Function:
    @classmethod
    def apply(cls, *inputs: Any) -> Any:
        ctx = FunctionCtx()
        with no_grad():
            forward_result = cls.forward(ctx, *inputs)

        outputs = _make_tensor_tuple(forward_result)

        input_requires_grad = any(
            x.requires_grad for x in inputs if isinstance(x, tinynn.Tensor)
        )
        if is_grad_enabled() and input_requires_grad:
            backward_fn = lambda *args: cls.backward(ctx, *args)
            grad_fn = FunctionNode(backward_fn, inputs, outputs)
            for output in outputs:
                output.requires_grad = True
                output.grad_fn = grad_fn

        return forward_result

    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: FunctionCtx, *args: Any) -> Any:
        raise NotImplementedError()


@no_grad()
def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
) -> None:
    # check tensors
    tensors = _make_tensor_tuple(tensors)
    assert all(tensor.requires_grad for tensor in tensors)

    # check grad_tensors
    if grad_tensors is None:
        if any(tensor.numel() != 1 for tensor in tensors):
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad_tensors = tuple(tinynn.ones_like(tensor) for tensor in tensors)
    grad_tensors = _make_tensor_tuple(grad_tensors)
    assert len(tensors) == len(grad_tensors)

    # compute dependencies (in-degrees)
    stack = [tensor.grad_fn for tensor in tensors if not tensor.is_leaf]
    visited = set()
    dependencies = {fn: 0 for fn in stack}
    while stack:
        fn = stack.pop()
        visited.add(fn)
        for input in fn.inputs:
            if isinstance(input, tinynn.Tensor) and input.grad_fn is not None:
                dependencies.setdefault(input.grad_fn, 0)
                dependencies[input.grad_fn] += 1
                if input.grad_fn not in visited:
                    stack.append(input.grad_fn)
    del visited

    # backward in topological order
    current_grads = {tensor: grad for tensor, grad in zip(tensors, grad_tensors)}
    stack = [fn for fn, dep in dependencies.items() if dep == 0]
    assert stack or not dependencies, "unexpected loop in forward graph"
    while stack:
        fn = stack.pop()
        dependencies.pop(fn)
        # collect gradients of output tensors
        grad_outputs = []
        for output in fn.outputs:
            grad_output = current_grads.get(output)
            if grad_output is None:
                grad_output = tinynn.zeros_like(output)
            grad_outputs.append(grad_output)
        # run backward to compute gradients of input tensors
        grad_inputs = _make_tensor_tuple(fn(*grad_outputs))
        assert len(fn.inputs) == len(grad_inputs)
        # accumulate gradients
        for input, grad_input in zip(fn.inputs, grad_inputs):
            if (
                isinstance(input, tinynn.Tensor)
                and input.requires_grad
                and grad_input is not None
            ):
                current_grads.setdefault(input, tinynn.zeros_like(input))
                current_grads[input] += grad_input
                if not input.is_leaf:
                    dependencies[input.grad_fn] -= 1
                    if dependencies[input.grad_fn] == 0:
                        stack.append(input.grad_fn)

    # accumulate the leaf gradients and discard the remaining
    for tensor, grad in current_grads.items():
        if tensor.is_leaf:
            if tensor.grad is None:
                tensor.grad = tinynn.zeros_like(tensor)
            tensor.grad += grad
