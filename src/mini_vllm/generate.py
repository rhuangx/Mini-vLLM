"""Single-request generation loops: the oracles every batched and paged loop is checked against."""

from __future__ import annotations

from collections.abc import Collection

import torch

from mini_vllm.kv_cache import KvFullCache
from mini_vllm.qwen3 import Qwen3Model
from mini_vllm.sampler import SamplingParams, sample

__all__ = ["GREEDY", "generate_with_kv_cache", "simple_generate"]

GREEDY = SamplingParams(temperature=0.0)


def _device(model: Qwen3Model) -> torch.device:
    return model.embedding.weight.device


def simple_generate(
    model: Qwen3Model,
    prompt: list[int],
    max_tokens: int,
    params: SamplingParams = GREEDY,
    eos_token_ids: Collection[int] = (),
) -> list[int]:
    """Recompute the whole sequence for every new token: quadratic, and needing no cache."""
    device = _device(model)
    tokens = list(prompt)
    for _ in range(max_tokens):
        logits = model(
            torch.tensor([tokens], dtype=torch.int32, device=device),
            torch.arange(len(tokens), device=device),
        )
        token = int(sample(logits[:, -1], params).item())
        tokens.append(token)
        if token in eos_token_ids:
            break
    return tokens[len(prompt) :]


def generate_with_kv_cache(
    model: Qwen3Model,
    prompt: list[int],
    max_tokens: int,
    params: SamplingParams = GREEDY,
    eos_token_ids: Collection[int] = (),
    prefill_chunk: int | None = None,
) -> list[int]:
    """Prefill the prompt, prefill_chunk tokens at a time, then forward one token per step."""
    device = _device(model)
    caches = [KvFullCache() for _ in model.layers]

    chunk = prefill_chunk or len(prompt)
    logits = None
    for start in range(0, len(prompt), chunk):
        piece = prompt[start : start + chunk]
        logits = model(
            torch.tensor([piece], dtype=torch.int32, device=device),
            torch.arange(start, start + len(piece), device=device),
            caches,
        )

    generated = []
    while True:
        token = int(sample(logits[:, -1], params).item())
        generated.append(token)
        if token in eos_token_ids or len(generated) == max_tokens:
            return generated
        # The new token sits at the position right after everything cached.
        logits = model(
            torch.tensor([[token]], dtype=torch.int32, device=device),
            torch.tensor([caches[0].offset], device=device),
            caches,
        )
