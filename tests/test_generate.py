"""generate.py: the two loops agree, and greedy decoding matches transformers on the real model."""

import pytest
import torch
from utils import with_and_without_cuda

from mini_vllm.generate import generate_with_kv_cache, simple_generate
from mini_vllm.reference.kv_cache import KvFullCache
from mini_vllm.utils.loader import from_hf

# Logits for Qwen3's top tokens sit in [16, 32), where one bf16 step is 0.125: a greedy choice
# decided by less than two steps is a tie that rounding alone can flip.
TIE_MARGIN = 0.25

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Paged attention stores the KV cache in",
    "Once upon a time, in a small village,",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM users",
    "To make a cup of tea, first",
    "import numpy as np\nx = np.",
    "The mitochondria is",
    "Translate to French: I would like a coffee.",
    "A haiku about autumn:",
    "The derivative of x^2 is",
    "Continuous batching improves throughput because",
    "Q: What is 17 * 23?\nA:",
    "The quick brown fox",
    "In 1969, Apollo 11",
]


def hf_greedy(model, prompt: list[int], max_tokens: int) -> list[int]:
    """Greedy continuation from a transformers model, one token at a time."""
    device = model.device
    ids = torch.tensor([prompt], device=device)
    continuation = []
    with torch.no_grad():
        for _ in range(max_tokens):
            token = int(torch.argmax(model(ids).logits[0, -1]))
            continuation.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=1)
    return continuation


@with_and_without_cuda
@pytest.mark.parametrize("prefill_chunk", [None, 1, 3])
def test_the_generation_loops_agree(tiny_qwen3, prefill_chunk, use_cuda):
    """The quadratic loop and the cached loop, token-identical; the cached loop also runs
    on the kernels, against the pure quadratic one."""
    prompt = torch.randint(0, 512, (7,), device="cuda").tolist()
    cached = generate_with_kv_cache(from_hf(tiny_qwen3, use_cuda), prompt, 12, prefill_chunk=prefill_chunk)
    assert cached == simple_generate(from_hf(tiny_qwen3), prompt, 12)


def test_generation_stops_at_an_eos_token(tiny_qwen3):
    model = from_hf(tiny_qwen3)
    prompt = torch.randint(0, 512, (7,), device="cuda").tolist()
    free = generate_with_kv_cache(model, prompt, 12)

    stop = free[3]
    expected = free[: free.index(stop) + 1]
    assert generate_with_kv_cache(model, prompt, 12, eos_token_ids={stop}) == expected
    assert simple_generate(model, prompt, 12, eos_token_ids={stop}) == expected


@with_and_without_cuda
def test_greedy_choices_match_transformers_away_from_bf16_ties(real_qwen3, use_cuda):
    """Teacher-forced along the HF greedy output, so both models see the same context at every
    step: wherever HF's top token wins by more than a bf16 tie, ours must pick it too. With
    the kernels, prompts over 8 tokens take flash_prefill and every decode step decode_attention."""
    reference, tokenizer = real_qwen3
    model = from_hf(reference, use_cuda)
    device = reference.device
    decided = 0

    for text in PROMPTS:
        prompt = tokenizer.encode(text)
        continuation = hf_greedy(reference, prompt, 32)

        # Our cached loop, fed HF's tokens instead of its own.
        caches = [KvFullCache() for _ in model.layers]
        logits = model(
            torch.tensor([prompt], device=device), torch.arange(len(prompt), device=device), caches
        )
        ours = [int(torch.argmax(logits[0, -1]))]
        for token in continuation[:-1]:
            logits = model(
                torch.tensor([[token]], device=device),
                torch.tensor([caches[0].offset], device=device),
                caches,
            )
            ours.append(int(torch.argmax(logits[0, -1])))

        # HF's own choice and margin at each of those steps, from one full forward pass.
        with torch.no_grad():
            theirs = reference(torch.tensor([prompt + continuation], device=device)).logits
        theirs = theirs[0, len(prompt) - 1 : -1].float()
        top_two = torch.sort(theirs, dim=-1).values[:, -2:]
        margins = (top_two[:, 1] - top_two[:, 0]).tolist()
        choices = torch.argmax(theirs, dim=-1).tolist()

        for step, margin in enumerate(margins):
            if margin > TIE_MARGIN:
                decided += 1
                assert ours[step] == choices[step], f"{text!r} step {step}: margin {margin:.3f}"

    # The tie exclusion must leave most steps decided, or this test would check nothing.
    assert decided > 0.8 * len(PROMPTS) * 32, decided
