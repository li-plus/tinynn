from typing import Any, Iterator, Tuple, Union

import numpy as np

import tinynn
import tinynn.nn.functional as F
from tinynn._tensor import Tensor


class Module:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def parameters(self) -> Iterator[Tensor]:
        for member in self.__dict__.values():
            if isinstance(member, Module):
                yield from member.parameters()
            elif isinstance(member, tinynn.Tensor):
                yield member

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for member in self.__dict__.values():
            if isinstance(member, Module):
                member.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(mode=False)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        stdv = 1 / np.sqrt(in_features)
        weight_init_data = np.random.uniform(-stdv, stdv, (out_features, in_features))
        self.weight = tinynn.tensor(weight_init_data, requires_grad=True)
        bias_init_data = np.random.uniform(-stdv, stdv, out_features)
        self.bias = tinynn.tensor(bias_init_data, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, p=self.p, training=self.training)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ) -> None:
        kh, kw = F._make_pair(kernel_size)

        stdv = 1 / np.sqrt(in_channels * kh * kw)
        weight_init_data = np.random.uniform(
            -stdv, stdv, size=(out_channels, in_channels, kh, kw)
        )
        self.weight = tinynn.tensor(weight_init_data, requires_grad=True)

        self.bias = None
        if bias:
            bias_init_data = np.random.uniform(-stdv, stdv, size=(out_channels,))
            self.bias = tinynn.tensor(bias_init_data, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight, self.bias)


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ) -> None:
        super().__init__()
        kh, kw = F._make_pair(kernel_size)

        stdv = 1 / np.sqrt(out_channels * kh * kw)
        weight_init_data = np.random.uniform(
            -stdv, stdv, size=(in_channels, out_channels, kh, kw)
        )
        self.weight = tinynn.tensor(weight_init_data, requires_grad=True)

        self.bias = None
        if bias:
            bias_init_data = np.random.uniform(-stdv, stdv, size=(out_channels,))
            self.bias = tinynn.tensor(bias_init_data, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv_transpose2d(input, self.weight, self.bias)


class MaxPool2d(Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(input, self.kernel_size)


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, reduction=self.reduction)
