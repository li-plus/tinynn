from numbers import Number
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import tinynn
from tinynn.autograd import Function, no_grad


class Tensor:
    def __init__(self, data, requires_grad: bool = False) -> None:
        self.data: np.ndarray = np.asarray(data)
        self.requires_grad: bool = requires_grad
        self.parents: Optional[Tuple[Tensor, ...]] = None
        self.grad_fn: Optional[Callable[[Tensor], Tuple[Tensor, ...]]] = None
        self.grad: Optional[Tensor] = None

    def __str__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    __repr__ = __str__

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def stride(self, dim: Optional[int] = None):
        return self.data.strides if dim is None else self.data.strides[dim]

    @property
    def is_leaf(self) -> bool:
        return not self.requires_grad or self.grad_fn is None

    def item(self) -> Any:
        return self.data.item()

    def numpy(self):
        return self.data

    def clone(self):
        return CloneFunction()(self)

    @property
    def T(self):
        return self.permute(tuple(range(self.ndim - 1, -1, -1)))

    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)

        if x.shape != y.shape:
            x, y = broadcast_tensors(x, y)

        z = AddFunction()(x, y)
        return z

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return NegativeFunction()(self)

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: Union["Tensor", Number]) -> "Tensor":
        return -self + other

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)
        if x.shape != y.shape:
            x, y = broadcast_tensors(x, y)

        z = MultiplyFunction()(x, y)
        return z

    __rmul__ = __mul__

    def reciprocal(self) -> "Tensor":
        return ReciprocalFunction()(self)

    def __truediv__(self, other: Union["Tensor", Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)
        return x * y.reciprocal()

    def __rtruediv__(self, other: Union["Tensor", Number]) -> "Tensor":
        return other * self.reciprocal()

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return MatmulFunction()(self, other)

    def __getitem__(self, index) -> "Tensor":
        return IndexFunction(index)(self)

    def sum(
        self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> "Tensor":
        return SumFunction(dim=dim, keepdim=keepdim)(self)

    def mean(
        self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> "Tensor":
        output = self.sum(dim, keepdim=keepdim)
        return output * (output.size / self.size)

    def permute(self, dims: Sequence[int]) -> "Tensor":
        return PermuteFunction(dims)(self)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        return ReshapeFunction(shape)(self)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        new_shape = np.squeeze(self.data, axis=dim).shape
        return self.reshape(new_shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if start_dim < 0:
            start_dim += self.ndim
        if end_dim < 0:
            end_dim += self.ndim

        if start_dim > end_dim:
            raise ValueError("start_dim cannot come after end_dim")
        if start_dim == end_dim:
            return self

        flat_size = np.prod(self.shape[start_dim : end_dim + 1])
        new_shape = self.shape[:start_dim] + (flat_size,) + self.shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def expand(self, sizes: Sequence[int]) -> "Tensor":
        return ExpandFunction(sizes)(self)

    def softmax(self, dim: int) -> "Tensor":
        return SoftmaxFunction(dim)(self)

    def log(self) -> "Tensor":
        return LogFunction()(self)

    def exp(self) -> "Tensor":
        return ExpFunction()(self)

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        return tensor(self.data.argmax(axis=dim, keepdims=keepdim))

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            idx = np.unravel_index(self.argmax().item(), shape=self.shape)
            output = self[idx]
        else:
            idx = list(np.indices(self.shape, sparse=True))
            idx[dim] = self.argmax(dim, keepdim=True)
            output = self[tuple(idx)]
            if not keepdim:
                output = output.squeeze(dim)
        return output

    def __hash__(self) -> int:
        return id(self)

    def eq(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data == other.data)

    __eq__ = eq

    def lt(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data < other.data)

    __lt__ = lt

    def le(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data <= other.data)

    __le__ = le

    def gt(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data > other.data)

    __gt__ = gt

    def ge(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data >= other.data)

    __ge__ = ge

    def ne(self, other: Union["Tensor", Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data != other.data)

    __ne__ = ne

    @no_grad()
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        assert self.requires_grad

        if gradient is None:
            assert self.size == 1
            gradient = tensor(np.array(1, dtype=np.float32))

        gradient.requires_grad = False

        stack = [(self, gradient)]
        while stack:
            variable, grad = stack.pop()

            if variable.is_leaf:
                # accumulate gradients for leaf nodes
                if variable.grad is None:
                    variable.grad = grad
                else:
                    variable.grad += grad
                continue

            # NOTE currently we only support single output op
            grads = variable.grad_fn(grad)
            if isinstance(grads, Tensor):
                grads = (grads,)
            assert len(grads) == len(variable.parents)
            for v, g in zip(variable.parents, grads):
                if g is not None and v.requires_grad:
                    stack.append((v, g))


def _ensure_tensor(x: Union[Tensor, Number]) -> Tensor:
    if isinstance(x, Number):
        x = tensor(x)
    assert isinstance(x, Tensor)
    return x


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def empty(size: Sequence[int], dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.empty(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def zeros(size: Sequence[int], dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.zeros(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def ones(size: Sequence[int], dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.ones(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def eye(
    n: int, m: Optional[int] = None, dtype=np.float32, requires_grad: bool = False
) -> Tensor:
    data = np.eye(n, m, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def rand(size, dtype=np.float32, requires_grad: bool = False):
    data = np.asarray(np.random.random(size), dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def randn(size, dtype=np.float32, requires_grad: bool = False):
    if isinstance(size, Number):
        size = (size,)
    data = np.asarray(np.random.randn(*size), dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def randint(low: int, high: int, size, dtype=np.int64, requires_grad: bool = False):
    data = np.random.randint(low, high, size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def broadcast_tensors(*tensors: Tensor) -> List[Tensor]:
    shapes = [x.shape for x in tensors]
    out_shape = np.broadcast_shapes(*shapes)
    out_tensors = [x.expand(out_shape) for x in tensors]
    return out_tensors


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return StackFunction(dim)(*tensors)


def where(
    condition: Tensor, x: Union[Tensor, Number], y: Union[Tensor, Number]
) -> Tensor:
    x, y = _ensure_tensor(x), _ensure_tensor(y)
    return WhereFunction(condition)(x, y)


class CloneFunction(Function):
    def forward(self, x: Tensor) -> Tensor:
        return tensor(x.data.copy())

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y


class AddFunction(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        return tinynn.tensor(x.data + y.data)

    def backward(self, grad_z: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_z, grad_z


class NegativeFunction(Function):
    def forward(self, x: Tensor) -> Tensor:
        return tensor(-x.data)

    def backward(self, grad_y: Tensor) -> Tensor:
        return -grad_y


class MultiplyFunction(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        self.inputs = (x, y)
        return tensor(x.data * y.data)

    def backward(self, grad_z: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = self.inputs
        return grad_z * y, grad_z * x


class ReciprocalFunction(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.y = tensor(1 / x.data)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        return -self.y * self.y * grad_y


class MatmulFunction(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.inputs = (x, y)
        return tensor(x.data @ y.data)

    def backward(self, grad_z: Tensor) -> Tensor:
        x, y = self.inputs
        return grad_z @ y.T, x.T @ grad_z


class IndexFunction(Function):
    def __init__(self, index) -> None:
        self.index = self._to_numpy_index(index)

    def forward(self, x: Tensor) -> Tensor:
        self.shape = x.shape
        return tensor(x.data[self.index])

    def backward(self, grad_y: Tensor) -> Tensor:
        # TODO: do not access data, implement setitem
        grad_x = zeros(self.shape)
        grad_x.data[self.index] = grad_y.data
        return grad_x

    @classmethod
    def _to_numpy_index(cls, index):
        if isinstance(index, Tensor):
            index = index.data
        elif isinstance(index, Sequence):
            index = type(index)(cls._to_numpy_index(x) for x in index)
        return index


class SumFunction(Function):
    def __init__(self, dim: Optional[Union[int, Sequence[int]]], keepdim: bool) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        self.shape = x.shape
        if self.dim is None:
            self.dim = tuple(range(x.ndim))

        return tensor(x.data.sum(axis=self.dim, keepdims=self.keepdim))

    def backward(self, grad_y: Tensor) -> Tensor:
        # TODO: do not access data directly
        grad_x = grad_y
        if not self.keepdim:
            grad_x.data = np.expand_dims(grad_x.data, axis=self.dim)
        grad_x = grad_x.expand(self.shape)
        return grad_x


class PermuteFunction(Function):
    def __init__(self, dims: Sequence[int]) -> None:
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return tensor(x.data.transpose(self.dims))

    def backward(self, grad_y: Tensor) -> Tensor:
        reversed_dims = np.empty(len(self.dims), dtype=np.int64)
        reversed_dims[list(self.dims)] = np.arange(len(self.dims))
        return grad_y.permute(reversed_dims)


class ReshapeFunction(Function):
    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        self.x_shape = x.shape
        return tensor(x.data.reshape(self.shape))

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y.reshape(self.x_shape)


class ExpandFunction(Function):
    def __init__(self, sizes: Sequence[int]) -> None:
        self.sizes = sizes

    def forward(self, x: Tensor) -> Tensor:
        self.x_shape = x.shape
        return tensor(np.broadcast_to(x.data, self.sizes))

    def backward(self, grad_y: Tensor) -> Tensor:
        grad_x = grad_y

        unsqueezed_dims = grad_x.ndim - len(self.x_shape)
        grad_x = grad_x.sum(tuple(range(unsqueezed_dims)))

        repeated_dims = []
        for dim in range(grad_x.ndim):
            if self.x_shape[dim] < grad_x.shape[dim]:
                assert self.x_shape[dim] == 1
                repeated_dims.append(dim)
        grad_x = grad_x.sum(tuple(repeated_dims), keepdim=True)

        return grad_x


class SoftmaxFunction(Function):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        numerator = (x - x.max()).exp()
        denominator = numerator.sum(dim=self.dim, keepdim=True)
        self.y = numerator / denominator
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        y_dy = self.y * grad_y
        grad_x = y_dy - self.y * y_dy.sum(dim=self.dim, keepdim=True)
        return grad_x


class LogFunction(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return tensor(np.log(x.data))

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y / self.x


class ExpFunction(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.y = tensor(np.exp(x.data))
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        return self.y * grad_y


class StackFunction(Function):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def forward(self, *tensors: Tensor) -> Tensor:
        self.num_tensors = len(tensors)
        return tensor(np.stack([x.data for x in tensors], axis=self.dim))

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, ...]:
        grad_input = tuple(
            tensor(data.squeeze(self.dim))
            for data in np.split(grad_output.data, self.num_tensors, axis=self.dim)
        )
        return grad_input


class WhereFunction(Function):
    def __init__(self, condition: Tensor) -> None:
        self.condition = condition

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return tensor(np.where(self.condition.data, x.data, y.data))

    def backward(self, grad_z: Tensor) -> Tensor:
        grad_x = where(self.condition, grad_z, 0)
        grad_y = where(self.condition, 0, grad_z)
        return grad_x, grad_y
