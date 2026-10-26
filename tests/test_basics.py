"""layers/activation.py and layers/linear.py against torch.nn."""

import pytest
import torch
import torch.nn.functional as F
from utils import DTYPES, assert_allclose, requires_cuda

from mini_vllm.layers.activation import silu, softmax, swiglu
from mini_vllm.layers.backends import get_backend
from mini_vllm.layers.linear import linear


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_matches_nn_linear(dtype):
    layer = torch.nn.Linear(48, 32, bias=True).to(dtype)
    x = torch.randn(3, 5, 48, dtype=dtype)

    assert_allclose(linear(x, layer.weight, layer.bias), layer(x))
    assert_allclose(linear(x, layer.weight), x @ layer.weight.T)


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_swiglu_match_nn_silu(dtype):
    gate = (4 * torch.randn(7, 64)).to(dtype)
    up = torch.randn(7, 64, dtype=dtype)

    assert_allclose(silu(gate), F.silu(gate))
    assert_allclose(swiglu(gate, up), F.silu(gate) * up)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [-1, 0])
def test_softmax_matches_torch_softmax(dtype, axis):
    x = (3 * torch.randn(6, 33)).to(dtype)
    assert_allclose(softmax(x, axis=axis), torch.softmax(x, dim=axis))


def test_softmax_is_stable_at_large_logits():
    x = torch.tensor([[1e4, 1e4 - 1.0, -1e4], [88.0, 89.0, 90.0]])
    out = softmax(x)

    assert not torch.any(torch.isnan(out))
    assert_allclose(out, torch.softmax(x, dim=-1))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_swiglu_kernel_matches_the_oracle(dtype):
    gate = (4 * torch.randn(5, 3072, device="cuda")).to(dtype)
    up = torch.randn(5, 3072, device="cuda", dtype=dtype)
    assert_allclose(get_backend("cuda").swiglu(gate, up), swiglu(gate, up))
    # A transposed view is made contiguous before the kernel reads it.
    assert_allclose(get_backend("cuda").swiglu(gate.T, up.T), swiglu(gate.T, up.T))
