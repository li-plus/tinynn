import itertools
import math
from collections import OrderedDict
from typing import Any, Iterator, Tuple, Union

import numpy as np

import tinynn
import tinynn.nn.functional as F
from tinynn._tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad)


class Buffer(Tensor):
    def __init__(self, data, requires_grad: bool = False) -> None:
        super().__init__(data, requires_grad)


class Module:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def parameters(self) -> Iterator[Tensor]:
        for _, v in self.named_parameters():
            yield v

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        for name, child in self.__dict__.items():
            if isinstance(child, Module):
                for sub_name, sub_param in child.named_parameters():
                    yield (f"{name}.{sub_name}", sub_param)
            elif isinstance(child, tinynn.nn.Parameter):
                yield (name, child)

    def buffers(self) -> Iterator[Tensor]:
        for _, v in self.named_buffers():
            yield v

    def named_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        for name, child in self.__dict__.items():
            if isinstance(child, Module):
                for sub_name, sub_buffer in child.named_buffers():
                    yield (f"{name}.{sub_name}", sub_buffer)
            elif isinstance(child, tinynn.nn.Buffer):
                yield (name, child)

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            {
                k: v.detach()
                for k, v in itertools.chain(
                    self.named_parameters(), self.named_buffers()
                )
            }
        )

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor]) -> None:
        state = self.state_dict()
        missing_keys = state.keys() - state_dict.keys()
        unexpected_keys = state_dict.keys() - state.keys()
        assert not (missing_keys or unexpected_keys)
        for k in state.keys():
            state[k].data[...] = state_dict[k].data

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
        stdv = 1 / math.sqrt(in_features)
        weight_init_data = np.random.uniform(
            -stdv, stdv, (out_features, in_features)
        ).astype(np.float32)
        self.weight = tinynn.nn.Parameter(weight_init_data)
        bias_init_data = np.random.uniform(-stdv, stdv, out_features).astype(np.float32)
        self.bias = tinynn.nn.Parameter(bias_init_data)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, p=self.p, training=self.training)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ) -> None:
        kh, kw = F._make_pair(kernel_size)

        stdv = 1 / math.sqrt(in_channels * kh * kw)
        weight_init_data = np.random.uniform(
            -stdv, stdv, size=(out_channels, in_channels, kh, kw)
        ).astype(np.float32)
        self.weight = tinynn.nn.Parameter(weight_init_data)

        self.bias = None
        if bias:
            bias_init_data = np.random.uniform(
                -stdv, stdv, size=(out_channels,)
            ).astype(np.float32)
            self.bias = tinynn.nn.Parameter(bias_init_data)

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

        stdv = 1 / math.sqrt(out_channels * kh * kw)
        weight_init_data = np.random.uniform(
            -stdv, stdv, size=(in_channels, out_channels, kh, kw)
        ).astype(np.float32)
        self.weight = tinynn.nn.Parameter(weight_init_data)

        self.bias = None
        if bias:
            bias_init_data = np.random.uniform(
                -stdv, stdv, size=(out_channels,)
            ).astype(np.float32)
            self.bias = tinynn.nn.Parameter(bias_init_data)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv_transpose2d(input, self.weight, self.bias)


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight = Parameter(tinynn.ones(num_features))
        self.bias = Parameter(tinynn.zeros(num_features))
        self.running_mean = Buffer(tinynn.zeros(num_features))
        self.running_var = Buffer(tinynn.ones(num_features))
        self.num_batches_tracked = Buffer(tinynn.zeros(1, dtype=np.int64))
        self.eps = eps
        self.momentum = momentum

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self.num_batches_tracked.data += 1  # TODO: inplace
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


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
