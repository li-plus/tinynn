import itertools
import random
import unittest
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

import tinynn


class TestTensor(unittest.TestCase):
    def _test_op(
        self, cases: List[Dict], fn: Callable, ref_fn: Optional[Callable] = None
    ):
        if ref_fn is None:
            ref_fn = fn

        for kwargs in cases:
            kwargs = kwargs.copy()
            shapes = kwargs.pop("shapes")

            test_tensors = []
            for s in shapes:
                if s is not None:
                    test_tensor = tinynn.randn(s, dtype=np.float64, requires_grad=True)
                else:
                    test_tensor = random.random()
                test_tensors.append(test_tensor)
            test_output = fn(*test_tensors, **kwargs)
            if test_output.requires_grad:
                test_output.backward(tinynn.ones(test_output.shape))

            ref_tensors = []
            for test_tensor in test_tensors:
                if isinstance(test_tensor, tinynn.Tensor):
                    ref_tensor = torch.tensor(test_tensor.data, requires_grad=True)
                else:
                    ref_tensor = test_tensor
                ref_tensors.append(ref_tensor)
            ref_output = ref_fn(*ref_tensors, **kwargs)
            if ref_output.requires_grad:
                ref_output.backward(torch.ones(ref_output.shape))

            assert np.allclose(ref_output.detach().numpy(), test_output.data)
            for ref_tensor, test_tensor in zip(ref_tensors, test_tensors):
                if (
                    isinstance(test_tensor, tinynn.Tensor)
                    and ref_tensor.grad is not None
                ):
                    assert np.allclose(ref_tensor.grad.numpy(), test_tensor.grad.data)
                    assert not test_tensor.grad.requires_grad

    def test_clone(self):
        cases = [dict(shapes=[(2, 3)])]
        self._test_op(cases, lambda x: x.clone())

    def test_transpose(self):
        cases = [
            dict(shapes=[(2, 3)]),
            dict(shapes=[(1, 3)]),
        ]
        self._test_op(cases, lambda x: x.T)

    def test_neg(self):
        cases = [
            dict(shapes=[(2, 3)]),
            dict(shapes=[(1,)]),
        ]
        self._test_op(cases, lambda x: -x)

    def test_elementwise(self):
        cases = [
            dict(shapes=[(2, 3), (2, 3)]),
            dict(shapes=[(2, 3), (3,)]),
            dict(shapes=[(2, 3), ()]),  # scalar tensor
            dict(shapes=[(2, 3), None]),  # number
            dict(shapes=[None, (2, 3)]),  # number
            dict(shapes=[(1, 3), (2, 1)]),  # broadcast on both tensors
        ]
        self._test_op(cases, lambda x, y: x + y)
        self._test_op(cases, lambda x, y: x - y)
        self._test_op(cases, lambda x, y: x * y)
        self._test_op(cases, lambda x, y: x / y)

    def test_matmul(self):
        cases = [
            dict(shapes=[(2, 3), (3, 4)]),
            dict(shapes=[(1, 3), (3, 1)]),
        ]
        self._test_op(cases, lambda x, y: x @ y)

    def test_getitem(self):
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
        self._test_op(cases, lambda x, index: x[index])

    def test_sum_mean(self):
        cases = []
        for dim in (None, 0, 1, 2, -1, -2, -3, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
            for keepdim in (False, True):
                c = dict(shapes=[(2, 3, 4)])
                if dim is not None:
                    c.update(dict(dim=dim, keepdim=keepdim))
                cases.append(c)
        self._test_op(cases, lambda x, **kwargs: x.sum(**kwargs))
        self._test_op(cases, lambda x, **kwargs: x.mean(**kwargs))

    def test_max(self):
        cases = []
        for dim in (None, 0, 1, 2, -1, -2, -3):
            for keepdim in (False, True):
                c = dict(shapes=[(2, 3, 4)])
                if dim is not None:
                    c.update(dict(dim=dim, keepdim=keepdim))
                cases.append(c)
        self._test_op(
            cases,
            lambda x, **kwargs: x.max(**kwargs),
            ref_fn=lambda x, **kwargs: x.max(**kwargs)[0] if kwargs else x.max(),
        )

    def test_reshape(self):
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
        self._test_op(cases, lambda x, new_shape: x.reshape(new_shape))

    def test_flatten(self):
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
        ]
        self._test_op(cases, lambda x, **kwargs: x.flatten(**kwargs))

    def test_permute(self):
        cases = [
            dict(shapes=[(2, 3, 4, 5)], dims=dims)
            for dims in itertools.permutations(range(4))
        ]
        self._test_op(cases, lambda x, dims: x.permute(dims=dims))

    def test_transform(self):
        cases = [
            dict(shapes=[(2, 3, 4, 5)], dim0=dim0, dim1=dim1)
            for dim0, dim1 in itertools.combinations(range(4), 2)
        ]
        self._test_op(cases, lambda x, **kwargs: x.transpose(**kwargs))

    def test_compare(self):
        cases = [
            dict(shapes=[(2, 3, 4), (2, 3, 4)]),
            dict(shapes=[(2, 3, 1), (1, 4)]),
            dict(shapes=[(2, 3, 4), None]),
            dict(shapes=[None, (2, 3, 4)]),
        ]
        self._test_op(cases, lambda x, y: x == y)
        self._test_op(cases, lambda x, y: x != y)
        self._test_op(cases, lambda x, y: x < y)
        self._test_op(cases, lambda x, y: x > y)
        self._test_op(cases, lambda x, y: x <= y)
        self._test_op(cases, lambda x, y: x >= y)

    def test_expand(self):
        cases = [
            dict(shapes=[(3, 4)], new_shape=(1, 3, 4)),
            dict(shapes=[(3, 4)], new_shape=(2, 3, 4)),
            dict(shapes=[(3, 4)], new_shape=(3, 3, 4)),
            dict(shapes=[(3, 4)], new_shape=(4, 3, 4)),
            dict(shapes=[(1, 4, 1)], new_shape=(3, 4, 5)),
            dict(shapes=[(1, 4, 1)], new_shape=(3, 4, 4, 4)),
        ]
        self._test_op(cases, lambda x, new_shape: x.expand(new_shape))

    def test_log(self):
        cases = [dict(shapes=[(3, 4)])]
        self._test_op(cases, lambda x: (x + 32).log())

    def test_exp(self):
        cases = [dict(shapes=[(3, 4)])]
        self._test_op(cases, lambda x: x.exp())

    def test_softmax(self):
        cases = [
            dict(shapes=[4], dim=0),
            dict(shapes=[(4, 5)], dim=0),
            dict(shapes=[(4, 5)], dim=1),
            dict(shapes=[(4, 5, 6)], dim=0),
            dict(shapes=[(4, 5, 6)], dim=1),
            dict(shapes=[(4, 5, 6)], dim=2),
        ]
        self._test_op(cases, lambda x, dim: x.softmax(dim=dim))

    def test_stack(self):
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
        self._test_op(
            cases,
            fn=lambda *tensors, dim: tinynn.stack(tensors, dim=dim),
            ref_fn=lambda *tensors, dim: torch.stack(tensors, dim=dim),
        )

    def test_where(self):
        cases = [
            dict(shapes=[(3, 4), (3, 4), (3, 4)]),
            dict(shapes=[(3, 4), None, (3, 4)]),
            dict(shapes=[(3, 4), (3, 4), None]),
        ]
        self._test_op(
            cases,
            fn=lambda cond, x, y: tinynn.where(cond > 0, x, y),
            ref_fn=lambda cond, x, y: torch.where(cond > 0, x, y),
        )

    def test_pad(self):
        cases = [
            dict(shapes=[(2, 3, 4)], pad=(1, 1)),
            dict(shapes=[(2, 3, 4)], pad=(2, 2, 1, 1)),
            dict(shapes=[(2, 3, 4)], pad=(0, 2, 1, 0)),
            dict(shapes=[(2, 3, 4)], pad=(2, 2, 1, 1, 4, 4)),
            dict(shapes=[(2, 3, 4)], pad=(1, 2, 2, 1, 4, 3)),
        ]
        self._test_op(
            cases,
            fn=lambda x, pad: tinynn.nn.functional.pad(x, pad),
            ref_fn=lambda x, pad: torch.nn.functional.pad(x, pad),
        )


class TestModule(unittest.TestCase):
    def test_module(self):
        class ToyModel(tinynn.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = tinynn.nn.Linear(2, 3)

            def forward(self, x):
                return self.fc(x)

        model = ToyModel()
        assert len(list(model.parameters())) == 2

    def test_relu(self):
        test_x = tinynn.randn((3, 4, 5), requires_grad=True)
        test_y = tinynn.nn.ReLU()(test_x)
        test_y.backward(tinynn.ones(test_y.shape))

        ref_x = torch.tensor(test_x.data, requires_grad=True)
        ref_y = torch.nn.ReLU()(ref_x)
        ref_y.backward(torch.ones(ref_y.shape))

        assert np.allclose(ref_y.detach().numpy(), test_y.data)
        assert np.allclose(ref_x.grad.numpy(), test_x.grad.data)
        assert not test_x.grad.requires_grad

    def test_dropout(self):
        for is_training in (True, False):
            test_model = tinynn.nn.Dropout(p=0.2)
            test_model.train(is_training)
            test_x = tinynn.randn((100, 200), requires_grad=True)
            test_y = test_model(test_x)
            test_y.backward(tinynn.ones(test_y.shape))
            keep_ratio = (test_y != 0).sum() / test_y.size
            assert 0.7 < keep_ratio < 0.9

            ref_model = torch.nn.Dropout(p=0.2)
            ref_model.train(is_training)
            ref_x = torch.tensor(test_x.data, requires_grad=True)
            ref_y = ref_model(ref_x)
            ref_y.backward(torch.ones(ref_y.shape))

            nonzero_idx = (test_y != 0).data & (ref_y != 0).numpy()
            assert np.allclose(
                ref_y.detach().numpy()[nonzero_idx], test_y.data[nonzero_idx]
            )
            assert np.allclose(
                ref_x.grad.numpy()[nonzero_idx], test_x.grad.data[nonzero_idx]
            )

    def test_conv2d(self):
        cases = [
            dict(
                model_name="Conv2d",
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                bias=True,
            ),
            dict(
                model_name="ConvTranspose2d",
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                bias=True,
            ),
        ]
        for kwargs in cases:
            name = kwargs.pop("model_name")

            ref_model = torch.nn.__dict__[name](**kwargs)
            ref_x = torch.randn((5, 8, 36, 36), requires_grad=True)
            ref_y = ref_model(ref_x)
            ref_y.backward(torch.ones(ref_y.shape))

            test_model = tinynn.nn.__dict__[name](**kwargs)
            test_model.weight.data = ref_model.weight.data.numpy()
            test_model.bias.data = ref_model.bias.data.numpy()
            test_x = tinynn.tensor(ref_x.detach().numpy(), requires_grad=True)
            test_y = test_model(test_x)
            test_y.backward(tinynn.ones(test_y.shape))

            assert np.allclose(ref_y.detach().numpy(), test_y.data, atol=1e-6)
            assert np.allclose(ref_x.grad.numpy(), test_x.grad.data, atol=1e-6)
            assert np.allclose(
                ref_model.weight.grad.numpy(), test_model.weight.grad.data, rtol=1e-3
            )
            assert np.allclose(ref_model.bias.grad.numpy(), test_model.bias.grad.data)

    def test_max_pool2d(self):
        cases = [
            dict(kernel_size=2),
            dict(kernel_size=3),
            dict(kernel_size=(2, 3)),
        ]
        for kwargs in cases:
            test_model = tinynn.nn.MaxPool2d(**kwargs)
            test_x = tinynn.randn((2, 3, 8, 8), requires_grad=True)
            test_y = test_model(test_x)
            test_y.backward(tinynn.ones(test_y.shape))

            ref_model = torch.nn.MaxPool2d(**kwargs)
            ref_x = torch.tensor(test_x.numpy(), requires_grad=True)
            ref_y = ref_model(ref_x)
            ref_y.backward(torch.ones(ref_y.shape))

            assert np.allclose(ref_y.detach().numpy(), test_y.data)
            assert np.allclose(ref_x.grad.numpy(), test_x.grad.data)

    def test_cross_entropy_loss(self):
        for reduction in ("none", "mean", "sum"):
            logits = tinynn.randn((100, 10), requires_grad=True)
            target = tinynn.randint(low=0, high=10, size=100)
            ce = tinynn.nn.CrossEntropyLoss(reduction=reduction)
            loss = ce(logits, target)
            loss.backward(tinynn.ones(loss.shape))

            ref_logits = torch.tensor(logits.data, requires_grad=True)
            ref_target = torch.tensor(target.data)
            ref_ce = torch.nn.CrossEntropyLoss(reduction=reduction)
            ref_loss = ref_ce(ref_logits, ref_target)
            ref_loss.backward(torch.ones(ref_loss.shape))

            assert np.allclose(ref_loss.detach().numpy(), loss.data)
            assert np.allclose(ref_logits.grad.numpy(), logits.grad.data)


class TestOptimizer(unittest.TestCase):
    def test_sgd(self):
        cases = [
            dict(lr=0.01),
            dict(lr=0.1, weight_decay=0.1),
            dict(lr=0.1, momentum=0.9),
            dict(lr=0.1, momentum=0.9, weight_decay=0.1),
        ]
        for kwargs in cases:
            init_weight = torch.randn(4, 3).numpy()
            init_bias = torch.randn(4).numpy()

            model = tinynn.nn.Linear(3, 4)
            model.weight.data = init_weight.copy()
            model.bias.data = init_bias.copy()
            optimizer = tinynn.optim.SGD(model.parameters(), **kwargs)

            ref_model = torch.nn.Linear(3, 4)
            ref_model.weight.data = torch.tensor(init_weight)
            ref_model.bias.data = torch.tensor(init_bias)
            ref_optimizer = torch.optim.SGD(ref_model.parameters(), **kwargs)

            for step in range(16):
                data = torch.randn(2, 3).numpy()

                loss = model(tinynn.tensor(data)).sum()
                loss.backward()
                if (step + 1) % 2 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                ref_loss = ref_model(torch.tensor(data)).sum()
                ref_loss.backward()
                if (step + 1) % 2 == 0:
                    ref_optimizer.step()
                    ref_optimizer.zero_grad()

                assert np.allclose(
                    ref_model.weight.detach().numpy(), model.weight.data, rtol=5e-4
                )
                assert np.allclose(
                    ref_model.bias.detach().numpy(), model.bias.data, rtol=1e-4
                )


class TestDataLoader(unittest.TestCase):
    def test(self):
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
                assert np.allclose(ref_inputs.numpy(), test_inputs.data)
                assert isinstance(test_target, tinynn.Tensor)
                assert np.allclose(ref_target.numpy(), test_target.data)
            assert batch_idx == len(test_loader) - 1
