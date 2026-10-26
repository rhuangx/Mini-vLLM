"""Comparison helpers and the tiny-model engine fixtures shared across tests."""

import pytest
import torch

from mini_vllm.config import EngineConfig
from mini_vllm.engine.llm_engine import LLM
from mini_vllm.utils.loader import from_hf

try:
    import mini_vllm_ext  # noqa: F401

    CUDA_BUILT = True
except (ImportError, OSError):  # OSError: the package is there but _ext.so is not built
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


class LetterTokenizer:
    """Enough of a tokenizer for the tiny model's 512-token vocabulary: token i is one letter."""

    def __init__(self, eos_token_ids=()):
        self.eos_token_ids = set(eos_token_ids)

    def encode(self, text):
        return [ord(c) % 512 for c in text]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def engine(tiny_qwen3, eos_token_ids=(), use_cuda=False, **config) -> LLM:
    config.setdefault("num_blocks", 64)
    config.setdefault("block_size", 4)
    return LLM(from_hf(tiny_qwen3, use_cuda), LetterTokenizer(eos_token_ids), EngineConfig(**config))


def assert_nothing_held(llm: LLM) -> None:
    assert not llm.scheduler.has_work
    assert llm.manager.pool.num_free == llm.kv.num_blocks, "blocks leaked"


PROMPTS = [[1, 2, 3], [5], [7] * 9, [2, 4], [9, 8, 7, 6, 5], [1], list(range(30, 70))]
