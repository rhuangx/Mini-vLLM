"""Tiny random Qwen3 and Llama models from transformers, on the GPU if there is one, and the
GPU backends this machine can run."""

import os

os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# H_q * D != E and G = 2, as in Qwen3-0.6B: both catch reshape bugs. Untied, so greedy
# decoding is not dominated by the current token's own embedding.
TINY = {
    "vocab_size": 512, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 128,
    "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
}
LLAMA3_ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 64}


def _gpu_backends() -> list[str]:
    if not torch.cuda.is_available():
        return []
    names = []
    try:
        import mini_vllm_ext  # noqa: F401

        names.append("cuda")
    except (ImportError, OSError):
        pass
    try:
        import triton  # noqa: F401

        names.append("triton")
    except ImportError:
        pass
    return names


GPU_BACKENDS = _gpu_backends()
BACKENDS = ["torch", *GPU_BACKENDS]


def tiny_hf(kind: str = "qwen3", **overrides):
    """A random fp32 transformers model, its norms perturbed so a missing norm would show."""
    if kind == "llama":
        model = LlamaForCausalLM(LlamaConfig(**{**TINY, "rope_scaling": LLAMA3_ROPE, **overrides}))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**{**TINY, "rope_theta": 1e6, **overrides}))
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if name.endswith("norm.weight"):
                weight.add_(0.1 * torch.randn_like(weight))
    return model.to(device=DEVICE, dtype=torch.float32).eval()


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(1234)


@pytest.fixture
def tiny_qwen3():
    return tiny_hf()
