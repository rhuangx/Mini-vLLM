"""Test environment and fixtures shared by every test."""

import os

# Before torch is imported: its Triton matmul overrides need Python.h, which this
# image does not ship, and the pure path is the cuBLAS matmul the tests measure.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from mini_vllm.models import DEFAULT_MODEL, Tokenizer, _eos_token_ids

SEED = 1234

# TF32 is off for fp32 matmul by default; a test that drifted onto it would miss 1e-5.
assert not torch.backends.cuda.matmul.allow_tf32

# Two properties of Qwen3-0.6B that catch reshape bugs: H_q * D != E, and G = 2.
TINY_QWEN3 = {
    "vocab_size": 512,
    "hidden_size": 64,           # E
    "num_hidden_layers": 2,
    "num_attention_heads": 4,    # H_q
    "num_key_value_heads": 2,    # H_k -> G = 2
    "head_dim": 32,              # D   -> H_q * D = 128 != E
    "intermediate_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    # Untied, unlike Qwen3-0.6B: a random tied model's logits are dominated by the current
    # token's own embedding, so greedy decoding repeats one token whatever the context and a
    # token-identity test proves nothing. Untied, 12-token runs hold 7-12 distinct tokens.
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def seeded():
    """Seed torch's global RNG before each test so a failure is reproducible."""
    torch.manual_seed(SEED)


def require_gpu() -> None:
    """Skip the calling test on a machine without a CUDA GPU, such as a Mac: the models and
    pools the engine tests build live on the device. The full suite runs on Modal."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


def make_tiny_qwen3(**overrides):
    """A randomly initialized fp32 Qwen3ForCausalLM on CUDA: the oracle our model must match."""
    require_gpu()
    cfg = Qwen3Config(**{**TINY_QWEN3, **overrides})
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).to(device="cuda", dtype=torch.float32)
    # nn.RMSNorm starts at all ones, which would hide a swapped or missing norm.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("norm.weight"):
                value.add_(0.1 * torch.randn_like(value))
    model.eval()
    return model


@pytest.fixture
def tiny_qwen3():
    return make_tiny_qwen3()


@pytest.fixture(scope="session")
def real_qwen3():
    """BF16 Qwen3-0.6B as (transformers model, wrapped tokenizer); skipped unless downloaded."""
    require_gpu()
    if not isinstance(try_to_load_from_cache(DEFAULT_MODEL, "model.safetensors"), str):
        pytest.skip(f"{DEFAULT_MODEL} is not in the Hugging Face cache")
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    return model, Tokenizer(tokenizer, _eos_token_ids(model, tokenizer))
