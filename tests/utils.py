"""Comparison helpers shared by every test."""

import pytest
import torch

try:
    import mini_vllm_ext  # noqa: F401

    CUDA_BUILT = True
except ImportError:
    CUDA_BUILT = False

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The shared numerical tolerances, keyed by dtype so no test picks its own threshold.
OP_TOLERANCES = {
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}


# Runs a test on the pure-torch path and, once the extension is built, on the kernels too.
_UNBUILT = pytest.mark.skipif(not CUDA_BUILT, reason="build src/extensions first")
with_and_without_cuda = pytest.mark.parametrize(
    "use_cuda", [False, pytest.param(True, id="cuda", marks=[pytest.mark.cuda, _UNBUILT])]
)


def requires_cuda(test):
    """A test that runs a CUDA kernel: marked `cuda`, and skipped until the extension is built."""
    return pytest.mark.cuda(_UNBUILT(test))


def assert_allclose(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare two arrays of the same shape and dtype at the tolerance that dtype implies."""
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"

    tolerance = OP_TOLERANCES[actual.dtype]
    if not torch.allclose(actual.float(), expected.float(), rtol=tolerance, atol=tolerance):
        worst = torch.max(torch.abs(actual.float() - expected.float())).item()
        raise AssertionError(f"max abs difference {worst:.3g} exceeds tolerance {tolerance:g}")
