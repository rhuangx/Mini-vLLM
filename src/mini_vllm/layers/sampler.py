"""Per-request sampling: greedy, temperature, top-k and top-p, vectorized over the batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mini_vllm.layers.activation import softmax
from mini_vllm.sampling_parameters import SamplingParams

__all__ = ["sample", "sampling_probabilities"]


# SamplingParams is frozen, so every caller can share one default instance.
_DEFAULT_SAMPLING = SamplingParams()


def _as_columns(
    params: SamplingParams | Sequence[SamplingParams],
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast per-request parameters into three B x 1 arrays."""
    rows = [params] * batch if isinstance(params, SamplingParams) else list(params)

    def column(values, dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)[:, None]

    return (
        column([row.temperature for row in rows], torch.float32),
        column([row.top_k for row in rows], torch.int32),
        column([row.top_p for row in rows], torch.float32),
    )


def sampling_probabilities(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams],
) -> torch.Tensor:
    """The fp32 [B, V] distribution each row will be sampled from, vectorized per row."""
    batch, vocab = logits.shape
    temperature, top_k, top_p = _as_columns(params, batch, logits.device)

    greedy = temperature == 0.0
    # Divide by 1.0 on greedy rows to stay branch-free; they are overwritten below.
    scaled = logits.float() / torch.where(greedy, 1.0, temperature)

    order = torch.argsort(-scaled, dim=-1)
    probabilities = softmax(torch.gather(scaled, -1, order), axis=-1)

    # top-k: keep ranks [0, k). k == 0 disables it.
    rank = torch.arange(vocab, device=logits.device)[None, :]
    keep = rank < torch.where(top_k == 0, vocab, top_k)

    # top-p: keep a token while the mass strictly before it is below p, boundary included.
    mass_before = torch.cumsum(probabilities, dim=-1) - probabilities
    keep = keep & (mass_before < top_p)

    # The most likely token is always kept, so no row is fully masked however small p is.
    keep = keep | (rank == 0)

    truncated = torch.where(keep, probabilities, torch.zeros_like(probabilities))
    truncated = truncated / torch.sum(truncated, dim=-1, keepdim=True)

    # Back to vocabulary order: argsort of a permutation is its inverse.
    result = torch.gather(truncated, -1, torch.argsort(order, dim=-1))

    one_hot = (rank == torch.argmax(logits, dim=-1, keepdim=True)).to(torch.float32)
    return torch.where(greedy, one_hot, result)


def sample(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams] = _DEFAULT_SAMPLING,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row: logits [B, V] -> int32 tokens [B]; greedy rows are exact."""
    # All greedy, the default: each row's argmax, with no sort over the vocabulary. The
    # distribution below would be one-hot at that same argmax, at ~50x the cost.
    rows = [params] if isinstance(params, SamplingParams) else params
    if all(row.is_greedy for row in rows):
        return torch.argmax(logits, dim=-1).to(torch.int32)
    probabilities = sampling_probabilities(logits, params)
    # Zeros are never drawn: truncated tokens and the zeros of a greedy one-hot.
    drawn = torch.multinomial(probabilities, 1, generator=generator)
    return drawn.squeeze(-1).to(torch.int32)
