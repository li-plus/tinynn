import itertools
import random
from copy import deepcopy
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pytest
import torch

import tinynn

# ==================== operator ====================


def _make_tensors(shapes) -> Tuple[List[tinynn.Tensor], List[torch.Tensor]]:
    test_tensors = []
    ref_tensors = []
    for s in shapes:
        if s is not None:
            test_tensor = tinynn.randn(s, requires_grad=True)
            ref_tensor = torch.tensor(test_tensor.detach().numpy(), requires_grad=True)
        else:
            test_tensor = random.random()
            ref_tensor = test_tensor
        test_tensors.append(test_tensor)
        ref_tensors.append(ref_tensor)
    return test_tensors, ref_tensors


def _check_op_forward_backward(case: Dict, fn: Callable, ref_fn: Callable):
    shapes = case.pop("shapes")
    test_inputs, ref_inputs = _make_tensors(shapes)

    exception = case.pop("exception", None)
    if exception is not None:
        with pytest.raises(exception):
            fn(*test_inputs, **case)
        with pytest.raises(exception):
            ref_fn(*ref_inputs, **case)
        return

    test_outputs = fn(*test_inputs, **case)
    if isinstance(test_outputs, tinynn.Tensor):
        test_outputs = (test_outputs,)
    test_variables = [x for x in test_outputs if x.requires_grad]
    test_grads = [tinynn.ones_like(v) for v in test_variables]
    if len(test_variables) == 1:
        test_variables[0].backward(test_grads[0])
    elif len(test_variables) > 1:
        tinynn.autograd.backward(test_variables, test_grads)

    ref_outputs = ref_fn(*ref_inputs, **case)
    if isinstance(ref_outputs, torch.Tensor):
        ref_outputs = (ref_outputs,)
    ref_variables = [x for x in ref_outputs if x.requires_grad]
    ref_grads = [torch.ones_like(v) for v in ref_variables]
    if len(ref_variables) == 1:
        ref_variables[0].backward(ref_grads[0])
    elif len(ref_variables) > 1:
        torch.autograd.backward(ref_variables, ref_grads)

    # check outputs
    for ref_out, test_out in zip(ref_outputs, test_outputs):
        assert ref_out.detach().numpy().dtype == test_out.detach().numpy().dtype
        assert np.allclose(ref_out.detach().numpy(), test_out.detach().numpy())

    # check input grads
    for ref_in, test_in in zip(ref_inputs, test_inputs):
        if isinstance(ref_in, torch.Tensor) and ref_in.grad is not None:
            assert ref_in.grad.numpy().dtype == test_in.grad.numpy().dtype
            assert np.allclose(ref_in.grad.numpy(), test_in.grad.numpy(), atol=1e-6)


def _check_op(cases: List[Dict], fn: Callable, ref_fn: Optional[Callable] = None):
    if ref_fn is None:
        ref_fn = fn

    for kwargs in cases:
        _check_op_forward_backward(kwargs.copy(), fn, ref_fn)


def test_result_type():
    DTYPE_TORCH2NP = {
        torch.bool: np.bool_,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }
    for ref_type1, test_type1 in DTYPE_TORCH2NP.items():
        test_tensor1 = tinynn.zeros(1, dtype=test_type1)
        ref_tensor1 = torch.zeros(1, dtype=ref_type1)
        for ref_type2, test_type2 in DTYPE_TORCH2NP.items():
            test_tensor2 = tinynn.zeros(1, dtype=test_type2)
            ref_tensor2 = torch.zeros(1, dtype=ref_type2)
            test_restype = tinynn.result_type(test_tensor1, test_tensor2)
            ref_restype = torch.result_type(ref_tensor1, ref_tensor2)
            assert DTYPE_TORCH2NP[ref_restype] == test_restype


def test_unary():
    cases = [dict(shapes=[(3, 4)])]
    _check_op(cases, lambda x: x)
    _check_op(cases, lambda x: x.detach())
    _check_op(cases, lambda x: x.clone())
    _check_op(cases, lambda x: x.T)

    # neg
    _check_op(cases, lambda x: -x)
    _check_op(cases, lambda x: x.neg())

    # abs
    _check_op(cases, lambda x: x.abs())
    _check_op(cases, lambda x: abs(x))

    # pow
    _check_op(cases, lambda x: x**4)
    _check_op(cases, lambda x: x.abs().pow(3.14))
    _check_op(cases, lambda x: x.square())
    _check_op(cases, lambda x: x.abs().sqrt())

    # log & exp
    _check_op(cases, lambda x: x.abs().log())
    _check_op(cases, lambda x: x.exp())

    # numpy
    with pytest.raises(RuntimeError):
        tinynn.randn((3, 4), requires_grad=True).numpy()
    with pytest.raises(RuntimeError):
        torch.randn((3, 4), requires_grad=True).numpy()
    x = tinynn.randn((3, 4))
    assert np.allclose(x.numpy(), x.data)

    # bool
    error_cases = [dict(shapes=[(3, 4)], exception=RuntimeError)]
    _check_op(error_cases, lambda x: bool(x))


def test_elementwise():
    lhs_cases = [
        dict(shapes=[(2, 3), (2, 3)]),
        dict(shapes=[(2, 3), (3,)]),
        dict(shapes=[(2, 3), ()]),  # scalar tensor
        dict(shapes=[(2, 3), None]),  # float scalar
        dict(shapes=[(2, 3)], y=1),  # integer scalar
        dict(shapes=[(1, 3), (2, 1)]),  # broadcast on both tensors
    ]
    rhs_cases = [
        dict(shapes=[None, (2, 3)]),  # float scalar
    ]
    cases = lhs_cases + rhs_cases

    self_cases = [dict(shapes=[(2, 3)])]

    _check_op(self_cases, lambda x: x + x)
    _check_op(cases, lambda x, y: x + y)
    _check_op(lhs_cases, lambda x, y: x.add(y))

    _check_op(self_cases, lambda x: x - x)
    _check_op(cases, lambda x, y: x - y)
    _check_op(lhs_cases, lambda x, y: x.sub(y))

    _check_op(self_cases, lambda x: x * x)
    _check_op(cases, lambda x, y: x * y)
    _check_op(lhs_cases, lambda x, y: x.mul(y))

    _check_op(self_cases, lambda x: x / x)
    _check_op(cases, lambda x, y: x / y)
    _check_op(lhs_cases, lambda x, y: x.div(y))


def test_matmul():
    cases = [
        dict(shapes=[(2, 3), (3, 4)]),
        dict(shapes=[(1, 3), (3, 1)]),
    ]
    _check_op(cases, lambda x, y: x @ y)


def test_getitem():
    cases = [
        dict(shapes=[(2, 3, 4)], index=0),
        dict(shapes=[(2, 3, 4)], index=slice(None)),
        dict(shapes=[(2, 3, 4)], index=slice(0)),
        dict(shapes=[(2, 3, 4)], index=slice(0, 1)),
        dict(shapes=[(2, 3, 4)], index=slice(0, 2, 2)),
        dict(shapes=[(2, 3, 4)], index=(slice(None), 0)),
        dict(shapes=[(2, 3, 4)], index=(slice(None), slice(0, 1))),
        dict(shapes=[(2, 3, 4)], index=(slice(None), slice(1), slice(2, 4))),
        dict(shapes=[(2, 3, 4)], index=(Ellipsis, slice(2, 4))),
    ]
    _check_op(cases, lambda x, index: x[index])


def test_sum_mean_var():
    cases = []
    for dim in (None, 0, 1, 2, -1, -2, -3, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
        for keepdim in (False, True):
            c = dict(shapes=[(2, 3, 4)])
            if dim is not None:
                c.update(dim=dim, keepdim=keepdim)
            cases.append(c)
    _check_op(cases, lambda x, **kwargs: x.sum(**kwargs))
    _check_op(cases, lambda x, **kwargs: x.mean(**kwargs))
    _check_op(cases, lambda x, **kwargs: x.var(**kwargs))
    _check_op(cases, lambda x, **kwargs: x.var(correction=0, **kwargs))


def test_max():
    cases = []
    for dim in (None, 0, 1, 2, -1, -2, -3):
        for keepdim in (False, True):
            c = dict(shapes=[(2, 3, 4)])
            if dim is not None:
                c.update(dim=dim, keepdim=keepdim)
            cases.append(c)
    _check_op(
        cases,
        lambda x, **kwargs: x.max(**kwargs),
    )


def test_reshape():
    cases = [
        dict(shapes=[(3, 4)], new_shape=(4, 3)),
        dict(shapes=[(3, 4)], new_shape=(3, 2, 2)),
        dict(shapes=[(3, 4)], new_shape=(2, 3, 2)),
        dict(shapes=[(3, 4)], new_shape=(2, 2, 3)),
        dict(shapes=[(3, 4)], new_shape=(12)),
        dict(shapes=[(3, 4)], new_shape=(3, 4, 1)),
        dict(shapes=[(3, 4)], new_shape=(1, 3, 4)),
        dict(shapes=[(3, 4)], new_shape=(1, 3, 1, 4, 1)),
        dict(shapes=[(3, 4)], new_shape=(-1,)),
        dict(shapes=[(3, 4)], new_shape=(-1, 2)),
    ]
    _check_op(cases, lambda x, new_shape: x.reshape(new_shape))


def test_flatten():
    cases = [
        dict(shapes=[(2, 3, 4, 5)]),
        dict(shapes=[(2, 3, 4, 5)], start_dim=1),
        dict(shapes=[(2, 3, 4, 5)], start_dim=2),
        dict(shapes=[(2, 3, 4, 5)], start_dim=3),
        dict(shapes=[(2, 3, 4, 5)], start_dim=-1),
        dict(shapes=[(2, 3, 4, 5)], start_dim=-2),
        dict(shapes=[(2, 3, 4, 5)], start_dim=-3),
        dict(shapes=[(2, 3, 4, 5)], start_dim=1, end_dim=2),
        dict(shapes=[(2, 3, 4, 5)], start_dim=-3, end_dim=-2),
        dict(shapes=[(2, 3, 4, 5)], exception=RuntimeError, start_dim=-1, end_dim=-2),
    ]
    _check_op(cases, lambda x, **kwargs: x.flatten(**kwargs))


def test_permute():
    cases = [
        dict(shapes=[(2, 3, 4, 5)], dims=dims)
        for dims in itertools.permutations(range(4))
    ]
    _check_op(cases, lambda x, dims: x.permute(dims=dims))


def test_transpose():
    cases = [
        dict(shapes=[(2, 3, 4, 5)], dim0=dim0, dim1=dim1)
        for dim0, dim1 in itertools.combinations(range(4), 2)
    ]
    _check_op(cases, lambda x, **kwargs: x.transpose(**kwargs))


def test_compare():
    cases = [
        dict(shapes=[(2, 3, 4), (2, 3, 4)]),
        dict(shapes=[(2, 3, 1), (1, 4)]),
        dict(shapes=[(2, 3, 4), None]),
        dict(shapes=[None, (2, 3, 4)]),
    ]
    _check_op(cases, lambda x, y: x == y)
    _check_op(cases, lambda x, y: x != y)
    _check_op(cases, lambda x, y: x < y)
    _check_op(cases, lambda x, y: x > y)
    _check_op(cases, lambda x, y: x <= y)
    _check_op(cases, lambda x, y: x >= y)


def test_expand():
    cases = [
        dict(shapes=[(3, 4)], new_shape=(1, 3, 4)),
        dict(shapes=[(3, 4)], new_shape=(2, 3, 4)),
        dict(shapes=[(3, 4)], new_shape=(3, 3, 4)),
        dict(shapes=[(3, 4)], new_shape=(4, 3, 4)),
        dict(shapes=[(1, 4, 1)], new_shape=(3, 4, 5)),
        dict(shapes=[(1, 4, 1)], new_shape=(3, 4, 4, 4)),
    ]
    _check_op(cases, lambda x, new_shape: x.expand(new_shape))


def test_softmax():
    cases = [
        dict(shapes=[4], dim=0),
        dict(shapes=[(4, 5)], dim=0),
        dict(shapes=[(4, 5)], dim=1),
        dict(shapes=[(4, 5, 6)], dim=0),
        dict(shapes=[(4, 5, 6)], dim=1),
        dict(shapes=[(4, 5, 6)], dim=2),
    ]
    _check_op(cases, lambda x, dim: x.softmax(dim=dim))


def test_split():
    cases = [
        dict(shapes=[(6, 10, 12)], split_size_or_sections=3, dim=0),
        dict(shapes=[(6, 10, 12)], split_size_or_sections=3, dim=1),
        dict(shapes=[(6, 10, 12)], split_size_or_sections=(2, 6, 4), dim=2),
    ]
    _check_op(
        cases,
        lambda x, split_size_or_sections, dim: x.split(split_size_or_sections, dim),
    )


def test_autograd():
    def split_fn(x):
        a, b, c, d, e = x.split(2, dim=1)
        return (a + b) * c + e - a

    cases = [dict(shapes=[(6, 10, 12)])]
    _check_op(cases, split_fn)


def test_cat_like():
    cases = [
        dict(shapes=[(3, 4, 5), (3, 4, 5)], dim=0),
        dict(shapes=[(3, 4, 5), (3, 4, 5)], dim=1),
        dict(shapes=[(3, 4, 5), (3, 4, 5)], dim=2),
        dict(shapes=[(3, 4, 5), (3, 4, 5)], dim=3),
        dict(shapes=[(3, 4, 5), (3, 4, 5), (3, 4, 5)], dim=0),
        dict(shapes=[(3, 4, 5), (3, 4, 5), (3, 4, 5)], dim=1),
        dict(shapes=[(3, 4, 5), (3, 4, 5), (3, 4, 5)], dim=2),
        dict(shapes=[(3, 4, 5), (3, 4, 5), (3, 4, 5)], dim=3),
    ]
    _check_op(
        cases,
        fn=lambda *tensors, dim: tinynn.stack(tensors, dim=dim),
        ref_fn=lambda *tensors, dim: torch.stack(tensors, dim=dim),
    )

    cases = [
        dict(shapes=[(3, 4, 5), (2, 4, 5), (4, 4, 5)], dim=0),
        dict(shapes=[(3, 2, 5), (3, 4, 5), (3, 5, 5)], dim=1),
        dict(shapes=[(3, 4, 5), (3, 4, 6), (3, 4, 7)], dim=2),
    ]
    _check_op(
        cases,
        fn=lambda *tensors, dim: tinynn.cat(tensors, dim=dim),
        ref_fn=lambda *tensors, dim: torch.cat(tensors, dim=dim),
    )


def test_where():
    cases = [
        dict(shapes=[(3, 4), (3, 4), (3, 4)]),
        dict(shapes=[(3, 4), None, (3, 4)]),
        dict(shapes=[(3, 4), (3, 4), None]),
    ]
    _check_op(
        cases,
        fn=lambda cond, x, y: tinynn.where(cond > 0, x, y),
        ref_fn=lambda cond, x, y: torch.where(cond > 0, x, y),
    )


def test_pad():
    cases = [
        dict(shapes=[(2, 3, 4)], pad=(1, 1)),
        dict(shapes=[(2, 3, 4)], pad=(2, 2, 1, 1)),
        dict(shapes=[(2, 3, 4)], pad=(0, 2, 1, 0)),
        dict(shapes=[(2, 3, 4)], pad=(2, 2, 1, 1, 4, 4)),
        dict(shapes=[(2, 3, 4)], pad=(1, 2, 2, 1, 4, 3)),
    ]
    _check_op(
        cases,
        fn=lambda x, pad: tinynn.nn.functional.pad(x, pad),
        ref_fn=lambda x, pad: torch.nn.functional.pad(x, pad),
    )


# ==================== module ====================


class TensorOf(NamedTuple):
    shape: Tuple[int, ...]
    dtype: np.dtype = np.float32


def _make_inputs(
    inputs: Dict,
) -> Tuple[Dict[str, tinynn.Tensor], ...]:
    test_inputs, ref_inputs = {}, {}
    for k, v in inputs.items():
        if isinstance(v, TensorOf):
            v = np.random.random(v.shape).astype(v.dtype) * 2 - 1

        if isinstance(v, np.ndarray):
            requires_grad = v.dtype != np.int64
            test_inputs[k] = tinynn.tensor(v, requires_grad=requires_grad)
            ref_inputs[k] = torch.tensor(v, requires_grad=requires_grad)
        else:
            test_inputs[k] = ref_inputs[k] = v

    return test_inputs, ref_inputs


def _check_state_and_grad(ref_model: torch.nn.Module, test_model: tinynn.nn.Module):
    # check state dict
    ref_state = ref_model.state_dict()
    test_state = test_model.state_dict()
    assert ref_state.keys() == test_state.keys()
    for k in ref_state:
        assert ref_state[k].numpy().dtype == test_state[k].numpy().dtype
        assert np.allclose(ref_state[k].numpy(), test_state[k].numpy())

    # check grad of all parameters
    ref_params = dict(ref_model.named_parameters())
    test_params = dict(test_model.named_parameters())
    assert ref_params.keys() == test_params.keys()
    for k in ref_params:
        if ref_params[k].grad is not None:
            assert ref_params[k].grad.numpy().dtype == test_params[k].grad.numpy().dtype
            assert np.allclose(
                ref_params[k].grad.numpy(),
                test_params[k].grad.numpy(),
                rtol=5e-4,
                atol=2e-5,
            )


def _copy_state_dict(test_model, ref_model):
    ref_state = {k: torch.tensor(v.numpy()) for k, v in test_model.state_dict().items()}
    ref_model.load_state_dict(ref_state)


def _check_module(
    ref_model: torch.nn.Module, test_model: tinynn.nn.Module, inputs: Dict
):
    # overwrite test_model parameters with ref_model
    _copy_state_dict(test_model, ref_model)
    _check_state_and_grad(ref_model, test_model)

    test_inputs, ref_inputs = _make_inputs(inputs)

    test_y = test_model(**test_inputs)
    test_y.backward(tinynn.ones_like(test_y))

    ref_y = ref_model(**ref_inputs)
    ref_y.backward(torch.ones_like(ref_y))

    # check output
    assert ref_y.detach().numpy().dtype == test_y.detach().numpy().dtype
    assert np.allclose(ref_y.detach().numpy(), test_y.detach().numpy(), atol=1e-6)

    # check grad of all tensor inputs
    for k, v in inputs.items():
        if isinstance(v, TensorOf):
            assert np.allclose(
                ref_inputs[k].grad.numpy(), test_inputs[k].grad.numpy(), atol=1e-6
            )

    # check grad of all model parameters
    _check_state_and_grad(ref_model, test_model)


def test_module():
    class TestModel(tinynn.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = tinynn.nn.Linear(2, 3)
            self.scale = tinynn.nn.Parameter(tinynn.ones(1))

        def forward(self, x):
            return self.fc(x) * self.scale

    class RefModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(2, 3)
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.fc(x) * self.scale

    test_model = TestModel()
    ref_model = RefModel()
    ref_model.fc.weight.data = torch.tensor(test_model.fc.weight.data)
    ref_model.fc.bias.data = torch.tensor(test_model.fc.bias.data)
    ref_model.scale.data = torch.tensor(test_model.scale.data)

    # warmup
    test_model(tinynn.randn((4, 2)))
    ref_model(torch.randn((4, 2)))

    # check named_parameters
    test_params = {k: v for k, v in test_model.named_parameters()}
    ref_params = {k: v for k, v in ref_model.named_parameters()}
    assert set(test_params.keys()) == set(ref_params.keys())
    for k in test_params.keys():
        assert np.allclose(
            test_params[k].detach().numpy(), ref_params[k].detach().numpy()
        )

    # check parameters
    assert len(list(test_model.parameters())) == len(list(ref_model.parameters()))

    # check state_dict
    test_sd = test_model.state_dict()
    ref_sd = ref_model.state_dict()
    assert set(test_sd.keys()) == set(ref_sd.keys())
    for k in test_sd.keys():
        assert np.allclose(test_sd[k].numpy(), ref_sd[k].numpy())

    # check load_state_dict
    new_state = deepcopy(test_model.state_dict())
    new_state["fc.weight"].data[0, 0] = 123
    assert test_model.state_dict()["fc.weight"][0, 0] != 123
    test_model.load_state_dict(new_state)
    assert test_model.state_dict()["fc.weight"][0, 0] == 123


def test_relu():
    ref_model = torch.nn.ReLU()
    test_model = tinynn.nn.ReLU()
    inputs = dict(input=TensorOf(shape=(3, 4, 5)))
    _check_module(ref_model, test_model, inputs)


def test_dropout():
    for is_training in (True, False):
        inputs = dict(input=TensorOf(shape=(100, 200)))
        test_inputs, ref_inputs = _make_inputs(inputs)

        test_model = tinynn.nn.Dropout(p=0.2).train(is_training)
        test_y = test_model(**test_inputs)
        test_y.backward(tinynn.ones_like(test_y))
        if is_training:
            keep_ratio = (test_y != 0).sum() / test_y.numel()
            assert 0.7 < keep_ratio < 0.9

        ref_model = torch.nn.Dropout(p=0.2).train(is_training)
        ref_y = ref_model(**ref_inputs)
        ref_y.backward(torch.ones_like(ref_y))

        nonzero_idx = (test_y != 0).numpy() & (ref_y != 0).numpy()
        assert np.allclose(
            ref_y.detach().numpy()[nonzero_idx],
            test_y.detach().numpy()[nonzero_idx],
        )
        assert np.allclose(
            ref_inputs["input"].grad.numpy()[nonzero_idx],
            test_inputs["input"].grad.numpy()[nonzero_idx],
        )


def test_conv2d():
    kwargs = dict(in_channels=8, out_channels=16, kernel_size=3, bias=True)
    inputs = dict(input=TensorOf(shape=(5, 8, 36, 36)))

    ref_model = torch.nn.Conv2d(**kwargs)
    test_model = tinynn.nn.Conv2d(**kwargs)
    _check_module(ref_model, test_model, inputs)

    ref_model = torch.nn.ConvTranspose2d(**kwargs)
    test_model = tinynn.nn.ConvTranspose2d(**kwargs)
    _check_module(ref_model, test_model, inputs)


def test_batch_norm2d():
    ref_model = torch.nn.BatchNorm2d(num_features=4)
    test_model = tinynn.nn.BatchNorm2d(num_features=4)
    inputs = dict(input=TensorOf(shape=(3, 4, 5, 6)))

    # training
    for _ in range(4):
        _check_module(ref_model, test_model, inputs)

    # eval
    ref_model.eval()
    test_model.eval()
    for _ in range(4):
        _check_module(ref_model, test_model, inputs)


def test_max_pool2d():
    kwargs_list = [
        dict(kernel_size=2),
        dict(kernel_size=3),
        dict(kernel_size=(2, 3)),
    ]
    for kwargs in kwargs_list:
        ref_model = torch.nn.MaxPool2d(**kwargs)
        test_model = tinynn.nn.MaxPool2d(**kwargs)
        inputs = dict(input=TensorOf(shape=(2, 3, 8, 8)))
        _check_module(ref_model, test_model, inputs)


def test_cross_entropy_loss():
    for reduction in ("none", "mean", "sum"):
        ref_model = torch.nn.CrossEntropyLoss(reduction=reduction)
        test_model = tinynn.nn.CrossEntropyLoss(reduction=reduction)
        inputs = dict(
            input=TensorOf(shape=(100, 10)),
            target=np.random.randint(0, 10, size=100, dtype=np.int64),
        )
        _check_module(ref_model, test_model, inputs)


# ==================== optimizer ====================


def _check_optimizer(
    inputs: Dict,
    test_model: tinynn.nn.Module,
    test_opt: tinynn.optim.Optimizer,
    ref_model: torch.nn.Module,
    ref_opt: torch.optim.Optimizer,
):
    _copy_state_dict(test_model, ref_model)
    test_inputs, ref_inputs = _make_inputs(inputs)

    for step in range(16):
        test_loss = test_model(**test_inputs).sum()
        test_loss.backward()
        if (step + 1) % 2 == 0:
            test_opt.step()
            test_opt.zero_grad()

        ref_loss = ref_model(**ref_inputs).sum()
        ref_loss.backward()
        if (step + 1) % 2 == 0:
            ref_opt.step()
            ref_opt.zero_grad()

        _check_state_and_grad(ref_model, test_model)


def test_sgd():
    kwargs_list = [
        dict(lr=0.01),
        dict(lr=0.1, weight_decay=0.1),
        dict(lr=0.1, momentum=0.9),
        dict(lr=0.1, momentum=0.9, weight_decay=0.1),
    ]
    for kwargs in kwargs_list:
        test_model = tinynn.nn.Linear(3, 4)
        ref_model = torch.nn.Linear(3, 4)

        test_opt = tinynn.optim.SGD(test_model.parameters(), **kwargs)
        ref_opt = torch.optim.SGD(ref_model.parameters(), **kwargs)

        inputs = dict(input=TensorOf(shape=(2, 3)))
        _check_optimizer(inputs, test_model, test_opt, ref_model, ref_opt)


def test_adam():
    kwargs_list = [
        dict(),
        dict(lr=0.01),
        dict(betas=(0.8, 0.888)),
        dict(eps=1e-5),
        dict(weight_decay=0.1),
    ]
    for kwargs in kwargs_list:
        test_model = tinynn.nn.Linear(3, 4)
        ref_model = torch.nn.Linear(3, 4)

        test_opt = tinynn.optim.Adam(test_model.parameters(), **kwargs)
        ref_opt = torch.optim.Adam(ref_model.parameters(), **kwargs)

        inputs = dict(input=TensorOf(shape=(2, 3)))
        _check_optimizer(inputs, test_model, test_opt, ref_model, ref_opt)


# ==================== dataloader ====================


def test_dataloader():
    cases = [
        dict(batch_size=2, drop_last=True),
        dict(batch_size=2, drop_last=False),
        dict(batch_size=3, drop_last=True),
        dict(batch_size=3, drop_last=False),
    ]
    num_samples = 10

    for c in cases:
        batch_size = c["batch_size"]
        drop_last = c["drop_last"]

        input_data = [(np.random.randn(28, 28), i) for i in range(num_samples)]

        test_loader = tinynn.utils.data.DataLoader(
            input_data, batch_size=batch_size, drop_last=drop_last
        )
        ref_loader = torch.utils.data.DataLoader(
            input_data, batch_size=batch_size, drop_last=drop_last
        )
        assert len(test_loader) == len(ref_loader)

        for batch_idx, (
            (ref_inputs, ref_target),
            (test_inputs, test_target),
        ) in enumerate(zip(ref_loader, test_loader)):
            assert isinstance(test_inputs, tinynn.Tensor)
            assert np.allclose(ref_inputs.numpy(), test_inputs.numpy())
            assert isinstance(test_target, tinynn.Tensor)
            assert np.allclose(ref_target.numpy(), test_target.numpy())
        assert batch_idx == len(test_loader) - 1
