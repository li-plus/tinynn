import math
from numbers import Number
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy._typing import _ShapeLike

import tinynn
from tinynn.autograd import Function, FunctionCtx, FunctionNode, backward

_Number = Union[int, float, complex]


class Tensor:
    def __init__(self, data, requires_grad: bool = False) -> None:
        if isinstance(data, Tensor):
            data = data.data
        dtype = np.float32 if isinstance(data, float) else None
        self.data = np.asarray(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad_fn: Optional[FunctionNode] = None
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

    def numel(self) -> int:
        return self.data.size

    def stride(self, dim: Optional[int] = None):
        return self.data.strides if dim is None else self.data.strides[dim]

    @property
    def is_leaf(self) -> bool:
        return not self.requires_grad or self.grad_fn is None

    def item(self) -> Any:
        return self.data.item()

    def numpy(self):
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return self.data

    def detach(self):
        return tensor(self.data)

    def clone(self):
        return CloneFunction.apply(self)

    @property
    def T(self):
        return self.permute(tuple(range(self.ndim - 1, -1, -1)))

    def add(self, other: Union["Tensor", _Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)
        x, y = _ensure_same_shape(x, y)
        z = AddFunction.apply(x, y)
        return z

    __radd__ = __add__ = add

    def neg(self) -> "Tensor":
        return NegativeFunction.apply(self)

    __neg__ = neg

    def sub(self, other: Union["Tensor", _Number]) -> "Tensor":
        return self + (-other)

    __sub__ = sub

    def __rsub__(self, other: Union["Tensor", _Number]) -> "Tensor":
        return -self + other

    def mul(self, other: Union["Tensor", _Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)
        x, y = _ensure_same_shape(x, y)
        z = MultiplyFunction.apply(x, y)
        return z

    __rmul__ = __mul__ = mul

    def reciprocal(self) -> "Tensor":
        return ReciprocalFunction.apply(self)

    def div(self, other: Union["Tensor", _Number]) -> "Tensor":
        x, y = self, _ensure_tensor(other)
        return x * y.reciprocal()

    __truediv__ = div

    def __rtruediv__(self, other: Union["Tensor", _Number]) -> "Tensor":
        return other * self.reciprocal()

    def abs(self) -> "Tensor":
        return where(self > 0, self, -self)

    __abs__ = abs

    def sqrt(self) -> "Tensor":
        return SqrtFunction.apply(self)

    def square(self) -> "Tensor":
        return SquareFunction.apply(self)

    def pow(self, exponent: float) -> "Tensor":
        return PowFunction.apply(self, exponent)

    __pow__ = pow

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return MatmulFunction.apply(self, other)

    def __getitem__(self, index) -> "Tensor":
        return IndexFunction.apply(self, index)

    def sum(
        self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> "Tensor":
        return SumFunction.apply(self, dim, keepdim)

    def mean(
        self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> "Tensor":
        output = self.sum(dim, keepdim=keepdim)
        n = self.numel() // output.numel()
        return output * (1 / n)

    def var(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        correction: int = 1,
        keepdim: bool = False,
    ) -> "Tensor":
        output = (
            (self - self.mean(dim, keepdim=True)).square().sum(dim, keepdim=keepdim)
        )
        n = self.numel() // output.numel()
        return output * (1 / (n - correction))

    def permute(self, dims: Sequence[int]) -> "Tensor":
        return PermuteFunction.apply(self, dims)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        return ReshapeFunction.apply(self, shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        return SqueezeFunction.apply(self, dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if start_dim < 0:
            start_dim += self.ndim
        if end_dim < 0:
            end_dim += self.ndim

        if start_dim > end_dim:
            raise RuntimeError(
                "flatten() has invalid args: start_dim cannot come after end_dim"
            )
        if start_dim == end_dim:
            return self

        flat_size = math.prod(self.shape[start_dim : end_dim + 1])
        new_shape = self.shape[:start_dim] + (flat_size,) + self.shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def expand(self, sizes: Sequence[int]) -> "Tensor":
        return ExpandFunction.apply(self, sizes)

    def softmax(self, dim: int) -> "Tensor":
        return SoftmaxFunction.apply(self, dim)

    def log(self) -> "Tensor":
        return LogFunction.apply(self)

    def exp(self) -> "Tensor":
        return ExpFunction.apply(self)

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        return tensor(self.data.argmax(axis=dim, keepdims=keepdim))

    def max(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        if dim is None:
            idx = np.unravel_index(self.argmax().item(), shape=self.shape)
            return self[idx]

        indices = self.argmax(dim=dim, keepdim=True)
        values = self.gather(dim=dim, index=indices)
        if not keepdim:
            indices = indices.squeeze(dim)
            values = values.squeeze(dim)
        return values, indices

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        return GatherFunction.apply(self, dim, index)

    def split(
        self, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0
    ) -> List["Tensor"]:
        return SplitFunction.apply(self, split_size_or_sections, dim)

    def __bool__(self) -> bool:
        if self.numel() > 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous"
            )
        return bool(self.item())

    def __hash__(self) -> int:
        return id(self)

    def eq(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data == other.data)

    __eq__ = eq

    def lt(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data < other.data)

    __lt__ = lt

    def le(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data <= other.data)

    __le__ = le

    def gt(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data > other.data)

    __gt__ = gt

    def ge(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data >= other.data)

    __ge__ = ge

    def ne(self, other: Union["Tensor", _Number]) -> "Tensor":
        other = _ensure_tensor(other)
        return tensor(self.data != other.data)

    __ne__ = ne

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        return backward(self, gradient)


_RESULT_TYPE_MAP = {
    ("float32", "int64"): np.float32,
    ("float32", "int32"): np.float32,
    ("float16", "int16"): np.float16,
    ("float16", "int32"): np.float16,
    ("float16", "int64"): np.float16,
}
_RESULT_TYPE_MAP.update({(t2, t1): v for (t1, t2), v in _RESULT_TYPE_MAP.items()})


def result_type(tensor1: Tensor, tensor2: Tensor) -> np.dtype:
    dtype = _RESULT_TYPE_MAP.get((tensor1.dtype.name, tensor2.dtype.name))
    if dtype is not None:
        return dtype
    return np.result_type(tensor1.data, tensor2.data)


def _ensure_tensor(x: Union[Tensor, _Number]) -> Tensor:
    if isinstance(x, Number):
        x = tensor(x)
    assert isinstance(x, Tensor)
    return x


def _ensure_same_shape(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    if x.shape != y.shape:
        x, y = broadcast_tensors(x, y)
    return x, y


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def empty(size: _ShapeLike, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.empty(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def zeros(size: _ShapeLike, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.zeros(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def zeros_like(
    input: Tensor, dtype: Optional[np.dtype] = None, requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    return zeros(input.shape, dtype=dtype, requires_grad=requires_grad)


def ones(size: _ShapeLike, dtype=np.float32, requires_grad: bool = False) -> Tensor:
    data = np.ones(size, dtype=dtype)
    return tensor(data, requires_grad=requires_grad)


def ones_like(
    input: Tensor, dtype: Optional[np.dtype] = None, requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    return ones(input.shape, dtype=dtype, requires_grad=requires_grad)


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


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return CatFunction.apply(dim, *tensors)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return StackFunction.apply(dim, *tensors)


def where(
    condition: Tensor, x: Union[Tensor, _Number], y: Union[Tensor, _Number]
) -> Tensor:
    x, y = _ensure_tensor(x), _ensure_tensor(y)
    return WhereFunction.apply(condition, x, y)


class CloneFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        return tensor(x.data.copy())

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        return grad_y


class AddFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        return tinynn.tensor(np.add(x.data, y.data, dtype=result_type(x, y)))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_z: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_z, grad_z


class NegativeFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        return tensor(-x.data)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        return -grad_y


class MultiplyFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        ctx.save_for_backward(x, y)
        return tensor(np.multiply(x.data, y.data, dtype=result_type(x, y)))

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_z: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x, y = ctx.saved_tensors
        grad_x = grad_z * y if x.requires_grad else None
        grad_y = grad_z * x if y.requires_grad else None
        return grad_x, grad_y


class ReciprocalFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        y = tensor(1 / x.data)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        return -y * y * grad_y


class SqrtFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        y = tensor(np.sqrt(x.data))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        return (0.5 / y) * grad_y


class SquareFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return tensor(np.square(x.data))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return 2 * x * grad_y


class PowFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, exponent: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.exponent = exponent
        return tensor(np.power(x.data, exponent))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        return ctx.exponent * x.pow(ctx.exponent - 1) * grad_y, None


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        return tensor(x.data @ y.data)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_z: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors
        return grad_z @ y.T, x.T @ grad_z


class IndexFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, index: Tensor) -> Tensor:
        ctx.index = IndexFunction._to_numpy_index(index)
        ctx.save_for_backward(x)
        return tensor(x.data[ctx.index])

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        # TODO: do not access data, implement setitem
        (x,) = ctx.saved_tensors
        grad_x = zeros_like(x)
        grad_x.data[ctx.index] = grad_y.data
        return grad_x, None

    @classmethod
    def _to_numpy_index(cls, index):
        if isinstance(index, Tensor):
            index = index.data
        elif isinstance(index, Sequence):
            index = type(index)(cls._to_numpy_index(x) for x in index)
        return index


class SumFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        dim: Optional[Union[int, Sequence[int]]],
        keepdim: bool,
    ) -> Tensor:
        ctx.x_shape = x.shape
        ctx.dim = dim if dim is not None else tuple(range(x.ndim))
        ctx.keepdim = keepdim
        return tensor(x.data.sum(axis=ctx.dim, keepdims=ctx.keepdim))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None, None]:
        # TODO: do not access data directly
        grad_x = grad_y
        if not ctx.keepdim:
            grad_x.data = np.expand_dims(grad_x.data, axis=ctx.dim)
        grad_x = grad_x.expand(ctx.x_shape)
        return grad_x, None, None


class PermuteFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, x: Tensor, dims: Sequence[int]
    ) -> Tuple[Tensor, None]:
        ctx.dims = dims
        return tensor(x.data.transpose(dims))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        reversed_dims = np.empty(len(ctx.dims), dtype=np.int64)
        reversed_dims[list(ctx.dims)] = np.arange(len(ctx.dims))
        return grad_y.permute(reversed_dims), None


class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, shape: Sequence[int]) -> Tensor:
        ctx.x_shape = x.shape
        return tensor(x.data.reshape(shape))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        return grad_y.reshape(ctx.x_shape), None


class SqueezeFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: Optional[int]) -> Tensor:
        ctx.x_shape = x.shape
        return tensor(x.data.squeeze(axis=dim))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        return grad_y.reshape(ctx.x_shape), None


class ExpandFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, sizes: Sequence[int]) -> Tensor:
        ctx.x_shape = x.shape
        return tensor(np.broadcast_to(x.data, sizes))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        grad_x = grad_y

        unsqueezed_dims = grad_x.ndim - len(ctx.x_shape)
        grad_x = grad_x.sum(tuple(range(unsqueezed_dims)))

        repeated_dims = []
        for dim in range(grad_x.ndim):
            if ctx.x_shape[dim] < grad_x.shape[dim]:
                assert ctx.x_shape[dim] == 1
                repeated_dims.append(dim)
        grad_x = grad_x.sum(tuple(repeated_dims), keepdim=True)

        return grad_x, None


class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        numerator = (x - x.max()).exp()
        denominator = numerator.sum(dim=dim, keepdim=True)
        y = numerator / denominator
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        (y,) = ctx.saved_tensors
        y_dy = y * grad_y
        grad_x = y_dy - y * y_dy.sum(dim=ctx.dim, keepdim=True)
        return grad_x, None


class LogFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return tensor(np.log(x.data))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_y / x


class ExpFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        y = tensor(np.exp(x.data))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        return y * grad_y


class GatherFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, dim: int, index: Tensor) -> Tensor:
        ctx.dim = dim
        ctx.index = index
        ctx.input_meta = dict(size=input.shape, dtype=input.dtype)
        return tensor(np.take_along_axis(input.data, index.data, axis=dim))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        grad_input = zeros(**ctx.input_meta)
        np.put_along_axis(
            grad_input.data, ctx.index.data, grad_output.data, axis=ctx.dim
        )
        return grad_input, None, None


class SplitFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        split_size_or_sections: Union[int, Sequence[int]],
        dim: int,
    ) -> Tuple[Tensor, ...]:
        ctx.dim = dim

        dim_size = x.shape[dim]
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            sections = [split_size] * (dim_size // split_size)
            remaining = dim_size - split_size * len(sections)
            if remaining > 0:
                sections.append(remaining)
        else:
            sections = split_size_or_sections

        indices = np.cumsum(sections[:-1])
        return tuple(tensor(data) for data in np.split(x.data, indices, axis=dim))

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_ys: Tensor) -> Tuple[Tensor, None, None]:
        return cat(grad_ys, dim=ctx.dim), None, None


class CatFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, dim: int, *tensors: Tensor) -> Tensor:
        ctx.dim = dim
        ctx.dim_sizes = [x.shape[dim] for x in tensors]
        return tensor(np.concatenate([x.data for x in tensors], axis=dim))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return None, *grad_output.split(ctx.dim_sizes, dim=ctx.dim)


class StackFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, dim: int, *tensors: Tensor) -> Tensor:
        ctx.dim = dim
        return tensor(np.stack([x.data for x in tensors], axis=dim))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_input = tuple(
            x.squeeze(ctx.dim) for x in grad_output.split(1, dim=ctx.dim)
        )
        return None, *grad_input


class WhereFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        ctx.condition = condition
        return tensor(np.where(condition.data, x.data, y.data))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_z: Tensor) -> Tuple[None, Tensor, Tensor]:
        grad_x = where(ctx.condition, grad_z, 0)
        grad_y = where(ctx.condition, 0, grad_z)
        return None, grad_x, grad_y
