from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import tinynn
from tinynn._tensor import Tensor
from tinynn.autograd import Function, FunctionCtx


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output += bias
    return output


def relu(input: Tensor) -> Tensor:
    return tinynn.where(input < 0, 0, input)


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return input
    to_drop = tinynn.rand(input.shape) < p
    output = tinynn.where(to_drop, 0, input) * (1 / (1 - p))
    return output


def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    output = Conv2dFunction.apply(input, weight)
    if bias is not None:
        output += bias[:, None, None]
    return output


def conv_transpose2d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    _, _, kh, kw = weight.shape
    padded_input = pad(input, (kw - 1, kw - 1, kh - 1, kh - 1))
    reversed_weight = weight.transpose(0, 1)[:, :, ::-1, ::-1]
    return conv2d(padded_input, reversed_weight, bias)


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    assert running_mean is not None and not running_mean.requires_grad
    assert running_var is not None and not running_var.requires_grad
    assert weight is not None and bias is not None

    def unsqz_stats(stats):
        return stats[tuple(None if i != 1 else slice(None) for i in range(input.ndim))]

    if training:
        dims = tuple(d for d in range(input.ndim) if d != 1)
        mean = input.mean(dims, keepdim=True)
        var = input.var(dims, correction=0, keepdim=True)

        # update running mean. TODO: inplace
        running_mean.data = (
            1 - momentum
        ) * running_mean.data + momentum * mean.data.squeeze()
        # update unbiased running var
        n = input.numel() // var.numel()
        running_var.data = (1 - momentum) * running_var.data + momentum * (
            n / (n - 1)
        ) * var.data.squeeze()
    else:
        mean = unsqz_stats(running_mean)
        var = unsqz_stats(running_var)

    norm_input = (input - mean) / (var + eps).sqrt()
    return norm_input * unsqz_stats(weight) + unsqz_stats(bias)


def max_pool2d(input: Tensor, kernel_size: Union[int, Tuple[int, int]]) -> Tensor:
    kernel_size = _make_pair(kernel_size)

    kh, kw = kernel_size
    n, c, ih, iw = input.shape
    oh = ih // kh  # output height
    ow = iw // kw  # output width

    output = (
        input[:, :, : oh * kh, : ow * kw]
        .reshape((n, c, oh, kh, ow, kw))
        .transpose(3, 4)
        .flatten(start_dim=-2)
        .max(dim=-1)[0]
    )
    return output


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    assert input.ndim == 2 and target.ndim == 1
    assert reduction in ("none", "mean", "sum")
    log_softmax = input.softmax(dim=1).log()
    num_classes = input.shape[-1]
    one_hot = tinynn.eye(num_classes, dtype=bool)[target]
    output = -log_softmax[one_hot]
    if reduction == "mean":
        output = output.mean()
    elif reduction == "sum":
        output = output.sum()
    return output


def pad(input: Tensor, pad: Sequence[int]) -> Tensor:
    return PadFunction.apply(input, pad)


class PadFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, pad: Sequence[int]) -> Tensor:
        assert len(pad) % 2 == 0, "padding must be even"
        pad = tuple((pad[i], pad[i + 1]) for i in range(len(pad) - 2, -1, -2))
        if len(pad) < x.ndim:
            pad = ((0, 0),) * (x.ndim - len(pad)) + pad
        ctx.pad = pad
        y = tinynn.tensor(np.pad(x.data, pad))
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: Tensor) -> Tuple[Tensor, None]:
        indices = tuple(
            slice(pad_before, s - pad_after)
            for s, (pad_before, pad_after) in zip(grad_y.shape, ctx.pad)
        )
        grad_x = grad_y[indices]
        return grad_x, None


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)

        _, _, kernel_height, kernel_width = weight.shape
        window_input_data = np.lib.stride_tricks.sliding_window_view(
            input.data, window_shape=(kernel_height, kernel_width), axis=(2, 3)
        )

        # window_input: [N, C, OH, OW, KH, KW], weight: [OC, C, KH, KW]
        output = tinynn.tensor(
            np.tensordot(
                window_input_data, weight.data, [(1, 4, 5), (1, 2, 3)]
            ).transpose((0, 3, 1, 2))
        )
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_tensors

        grad_input = None
        if input.requires_grad:
            grad_input = conv_transpose2d(grad_output, weight)

        grad_weight = None
        if weight.requires_grad:
            grad_weight = conv2d(
                input.transpose(0, 1), grad_output.transpose(0, 1)
            ).transpose(0, 1)

        return grad_input, grad_weight


def _make_pair(scalar_or_pair):
    if isinstance(scalar_or_pair, Number):
        pair = (scalar_or_pair, scalar_or_pair)
    else:
        pair = tuple(scalar_or_pair)
    assert len(pair) == 2
    return pair
