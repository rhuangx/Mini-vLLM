"""Grouped-query attention, dense and over a paged KV cache, and the fp8 pages it reads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mini_vllm.layers.activation import softmax

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import HostLists

__all__ = [
    "FP8_MAX",
    "causal_mask",
    "dequantize_fp8",
    "gather_pages",
    "paged_attention",
    "quantize_fp8",
    "scaled_dot_product_attention_grouped",
]


def causal_mask(
    query_len: int,
    source_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """An additive [L, S] mask: query i may attend to key j <= S - L + i."""
    # The k = S - L offset is what makes a chunk's rectangular mask line up with its history.
    # Bottom-right aligned, which is not what SDPA's is_causal does.
    allowed = torch.tril(
        torch.ones((query_len, source_len), dtype=torch.bool, device=device),
        diagonal=source_len - query_len,
    )
    zero = torch.zeros((), dtype=dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    return torch.where(allowed, zero, neg_inf)


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """query is B x H_q x L x D, key and value are B x H_k x S x D; mask is "causal" or
    additive and broadcastable to B x H_q x L x S."""
    *batch, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads, source_len = key.shape[-3], key.shape[-2]
    group_size = num_query_heads // num_kv_heads
    dtype = query.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Scores, softmax and the weighted sum stay fp32: a bf16 score near 10 is only good to
    # 0.06, which is enough to flip Qwen3-0.6B's greedy choice.
    query, key, value = (x.float() for x in (query, key, value))

    # Split the query heads into (kv_head, group) so K and V broadcast over the group
    # instead of being repeated G times.
    query = query.reshape(*batch, num_kv_heads, group_size, query_len, head_dim)
    key = key.unsqueeze(-3)
    value = value.unsqueeze(-3)

    scores = (query @ key.transpose(-2, -1)) * scale

    if isinstance(mask, str):
        scores = scores + causal_mask(query_len, source_len, scores.dtype, scores.device)
    elif mask is not None:
        mask = torch.broadcast_to(mask, (*batch, num_query_heads, query_len, source_len))
        scores = scores + mask.reshape(*batch, num_kv_heads, group_size, query_len, source_len)

    out = softmax(scores, axis=-1) @ value
    return out.reshape(*batch, num_query_heads, query_len, head_dim).to(dtype)


def gather_pages(
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    k_scale: float,
    v_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence's keys and values, S x H_k x D, read out of its pages and dequantized if
    they are fp8. block_table is that sequence's row of the device table."""
    block_size, num_kv_heads, head_dim = key_pages.shape[1:]
    blocks_used = -(-context_len // block_size)
    # Slice the device table: gathering from a host list would copy it back to the device.
    ids = block_table[:blocks_used].long()
    keys = key_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]
    values = value_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]

    # Dequantize only the pages this sequence reads, never the whole pool.
    if key_pages.dtype == torch.uint8:
        keys = dequantize_fp8(keys, k_scale, dtype)
        values = dequantize_fp8(values, v_scale, dtype)
    return keys, values


def paged_attention(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    host: HostLists | None = None,
) -> torch.Tensor:
    """Causal grouped attention over a paged cache, gathering each sequence's pages first.

    query is T x H_q x D, the pages are num_blocks x P x H_k x D (uint8 for fp8), and the
    int32 metadata is block_tables (N x max_blocks, -1 padded), cu_seqlens_q (N + 1) and
    context_lens (N).

    host, when the caller already has them, holds the batch's metadata as Python lists, read
    instead of .tolist(), which would sync the device once per list.
    Direct calls omit it and fall back.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if host is None:
        starts, lengths = cu_seqlens_q.tolist(), context_lens.tolist()
    else:
        starts, lengths = host.offsets, host.context_lens

    outputs = []
    for index, context_len in enumerate(lengths):
        start, end = starts[index], starts[index + 1]
        keys, values = gather_pages(
            key_pages, value_pages, block_tables[index], context_len, k_scale, v_scale, query.dtype
        )
        attended = scaled_dot_product_attention_grouped(
            query[start:end].transpose(0, 1),  # H_q x L x D
            keys.transpose(0, 1),              # H_k x S x D
            values.transpose(0, 1),
            scale=scale,
            mask="causal",
        )
        outputs.append(attended.transpose(0, 1))

    return torch.cat(outputs, dim=0)


# The largest finite e4m3 value. Torch's cast turns anything past it into NaN, so quantize
# saturates first, which is what mx.to_fp8 used to do.
FP8_MAX = 448.0


def quantize_fp8(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """x / scale as e4m3 bits in uint8: half the bytes of bf16, 3 mantissa bits."""
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_fp8(x: torch.Tensor, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The inverse of quantize_fp8: decode the uint8 bits, multiply the scale back in."""
    return (x.view(torch.float8_e4m3fn).float() * scale).to(dtype)
