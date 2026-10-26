"""Grouped-query attention, dense and over a paged KV cache."""

from __future__ import annotations

import math

import torch

from mini_vllm.layers.activation import softmax
from mini_vllm.layers.quantization import dequantize_fp8

__all__ = ["causal_mask", "paged_attention", "scaled_dot_product_attention_grouped"]


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
    use_cuda: bool = False,
) -> torch.Tensor:
    """query is B x H_q x L x D, key and value are B x H_k x S x D; mask is "causal" or
    additive and broadcastable to B x H_q x L x S.

    With use_cuda, a causal or unmasked call goes to decode_attention for a few query
    tokens, or to flash_prefill for a bf16, D = 128 prefill; anything else runs below.
    """
    *batch, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads, source_len = key.shape[-3], key.shape[-2]
    group_size = num_query_heads // num_kv_heads
    dtype = query.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if use_cuda and not isinstance(mask, torch.Tensor):
        import mini_vllm_ext

        causal = mask == "causal"
        if query_len <= 8:
            return mini_vllm_ext.decode_attention(query, key, value, scale, causal)
        if query.dtype == torch.bfloat16 and head_dim == 128:
            return mini_vllm_ext.flash_prefill(query, key, value, scale, causal)

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
    use_cuda: bool = False,
    host: tuple[list[list[int]], list[int], list[int]] | None = None,
) -> torch.Tensor:
    """Causal grouped attention over a paged cache, gathering each sequence's pages first.

    query is T x H_q x D, the pages are num_blocks x P x H_k x D (uint8 for fp8), and the
    int32 metadata is block_tables (N x max_blocks, -1 padded), cu_seqlens_q (N + 1) and
    context_lens (N).

    host, when the caller already has them, is (block tables, query starts, context lengths)
    as Python lists. The pure path and the kernel routing read those instead of .tolist(),
    which would sync the device once per list. Direct calls omit it and fall back.

    With use_cuda, the sequences computing at most 8 tokens (decodes) go to one paged
    kernel that walks the block tables itself. It gives each query token a thread block, so a
    long chunk would re-read its whole context once per token: those gather their pages and
    take the dense kernels instead.
    """
    is_fp8 = key_pages.dtype == torch.uint8
    if scale is None:
        scale = query.shape[-1] ** -0.5

    if host is None:
        starts = cu_seqlens_q.tolist()
        lengths = context_lens.tolist()
    else:
        starts, lengths = host[1], host[2]

    short = [i for i in range(len(lengths)) if starts[i + 1] - starts[i] <= 8] if use_cuda else []
    pieces = {}
    if short:
        import mini_vllm_ext

        if len(short) == len(lengths):
            pages = (key_pages, value_pages, block_tables, cu_seqlens_q, context_lens)
            return mini_vllm_ext.paged_attention(query, *pages, scale, k_scale, v_scale)
        # A mixed batch: the decodes as one sub-batch, whose rows are split back out below.
        rows = [row for i in short for row in range(starts[i], starts[i + 1])]
        sub_starts = [0]
        for i in short:
            sub_starts.append(sub_starts[-1] + starts[i + 1] - starts[i])
        chosen = torch.tensor(short, dtype=torch.long, device=block_tables.device)
        attended = mini_vllm_ext.paged_attention(
            query[torch.tensor(rows, dtype=torch.long, device=query.device)],
            key_pages,
            value_pages,
            block_tables[chosen],
            torch.tensor(sub_starts, dtype=torch.int32, device=query.device),
            context_lens[chosen],
            scale,
            k_scale,
            v_scale,
        )
        pieces = {i: attended[sub_starts[j] : sub_starts[j + 1]] for j, i in enumerate(short)}

    block_size, num_kv_heads, head_dim = key_pages.shape[1:]
    outputs = []

    for index, context_len in enumerate(lengths):
        if index in pieces:
            outputs.append(pieces[index])
            continue
        start, end = starts[index], starts[index + 1]
        blocks_used = -(-context_len // block_size)
        # Slice the device table: gathering from a host list would copy it back to the device.
        ids = block_tables[index, :blocks_used].long()
        keys = key_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]
        values = value_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]

        # Dequantize only the pages this sequence reads, never the whole pool.
        if is_fp8:
            keys = dequantize_fp8(keys, k_scale, query.dtype)
            values = dequantize_fp8(values, v_scale, query.dtype)

        attended = scaled_dot_product_attention_grouped(
            query[start:end].transpose(0, 1),  # H_q x L x D
            keys.transpose(0, 1),              # H_k x S x D
            values.transpose(0, 1),
            scale=scale,
            mask="causal",
            use_cuda=use_cuda,
        )
        outputs.append(attended.transpose(0, 1))

    return torch.cat(outputs, dim=0)
