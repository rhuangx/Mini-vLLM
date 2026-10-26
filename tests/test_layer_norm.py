"""layers/layernorm.py against F.rms_norm."""

import pytest
import torch
import torch.nn.functional as F
from utils import DTYPES, assert_allclose, requires_cuda

from mini_vllm.layers.layernorm import RMSNorm


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [128, 1024])  # Qwen3-0.6B's per-head q/k norm and its hidden size
def test_rms_norm_matches_torch(dtype, dim):
    weight = (1 + 0.1 * torch.randn(dim)).to(dtype)
    x = (3 * torch.randn(2, 5, dim)).to(dtype)

    assert_allclose(RMSNorm(dim, weight)(x), F.rms_norm(x, (dim,), weight, 1e-6))


def test_rms_norm_casts_back_before_the_weight_multiply():
    weight = (1 + 0.1 * torch.randn(64)).bfloat16()
    x = torch.randn(4, 64).bfloat16()

    x32 = x.float()
    normalized = x32 * torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-6)
    expected = weight * normalized.bfloat16()

    assert torch.equal(RMSNorm(64, weight)(x), expected)


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [128, 1024, 3000])  # a q/k head, the hidden size, a strided row
def test_the_rms_norm_kernel_matches_the_oracle(dtype, dim):
    """In bf16 and fp16 the kernel is bit-exact, which is what shows it rounds before the
    weight multiply as the oracle does. In fp32 it sums the squares in a different order,
    so a fraction of outputs move by an ulp."""
    weight = (1 + 0.1 * torch.randn(dim, device="cuda")).to(dtype)
    x = (3 * torch.randn(4, 3, dim, device="cuda")).to(dtype)
    norm, cuda = RMSNorm(dim, weight), RMSNorm(dim, weight, use_cuda=True)
    for rows in (x, x.transpose(0, 1)):  # contiguous, then a strided view
        if dtype == torch.float32:
            assert_allclose(cuda(rows), norm(rows))
        else:
            assert torch.equal(cuda(rows), norm(rows))
