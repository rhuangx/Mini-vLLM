"""The CUDA backend: each operator on its src/extensions kernel, and the pure path for the
shapes a kernel does not take."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mini_vllm.layers.attention import gather_pages
from mini_vllm.layers.backends import SHORT_QUERY
from mini_vllm.layers.backends.torch_backend import TorchBackend

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import HostLists

__all__ = ["CudaBackend", "split_short"]


def split_short(
    cu_seqlens_q: torch.Tensor, host: HostLists | None, num_short: tuple[int, int], max_query_len: int | None
) -> tuple[int, int, bool]:
    """(sequences, tokens) of the leading run of short sequences, and whether that is all of
    them. The engine puts decodes first and says so; a direct call is worked out here."""
    if max_query_len is not None and max_query_len <= SHORT_QUERY:
        return len(cu_seqlens_q) - 1, -1, True
    if num_short != (0, 0) or max_query_len is not None:
        return num_short[0], num_short[1], False
    starts = cu_seqlens_q.tolist() if host is None else host.offsets
    count = 0
    while count < len(starts) - 1 and starts[count + 1] - starts[count] <= SHORT_QUERY:
        count += 1
    return count, starts[count], count == len(starts) - 1


class CudaBackend(TorchBackend):
    name = "cuda"
    supports_graphs = True

    def __init__(self) -> None:
        import mini_vllm_ext  # raises here when src/extensions is not built

        self.ext = mini_vllm_ext

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        return self.ext.silu_and_mul(x)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return self.ext.rms_norm(x, weight, eps)

    def rope(
        self, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return self.ext.rope(x, positions, cos, sin)

    def dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """A causal or unmasked call goes to decode_attention for a few query tokens, or to
        flash_prefill for a bf16, D = 128 prefill; anything else runs on the pure path."""
        if isinstance(mask, torch.Tensor):
            return super().dense_attention(query, key, value, scale, mask)
        head_dim = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)  # as the pure path computes it, to the bit
        causal = mask == "causal"
        if query.shape[-2] <= SHORT_QUERY:
            return self.ext.decode_attention(query, key, value, scale, causal)
        if query.dtype == torch.bfloat16 and head_dim == 128:
            return self.ext.flash_prefill(query, key, value, scale, causal)
        return super().dense_attention(query, key, value, scale, mask)

    def paged_attention(
        self,
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
        num_short: tuple[int, int] = (0, 0),
        max_query_len: int | None = None,
    ) -> torch.Tensor:
        """The leading short sequences (decodes) go to one paged kernel that walks the block
        tables itself, as contiguous slices. It gives each query token a thread block, so a
        long chunk would re-read its whole context once per token: those gather their pages
        and take dense_attention instead."""
        if scale is None:
            scale = query.shape[-1] ** -0.5
        short_seqs, short_tokens, all_short = split_short(cu_seqlens_q, host, num_short, max_query_len)
        if all_short:
            pages = (key_pages, value_pages, block_tables, cu_seqlens_q, context_lens)
            return self.ext.paged_attention(query, *pages, scale, k_scale, v_scale)

        out = torch.empty_like(query)
        if short_seqs:
            out[:short_tokens] = self.ext.paged_attention(
                query[:short_tokens], key_pages, value_pages, block_tables[:short_seqs],
                cu_seqlens_q[: short_seqs + 1], context_lens[:short_seqs], scale, k_scale, v_scale,
            )
        if host is None:
            starts, lengths = cu_seqlens_q.tolist(), context_lens.tolist()
        else:
            starts, lengths = host.offsets, host.context_lens
        for index in range(short_seqs, len(lengths)):
            start, end = starts[index], starts[index + 1]
            keys, values = gather_pages(
                key_pages, value_pages, block_tables[index], lengths[index], k_scale, v_scale, query.dtype
            )
            attended = self.dense_attention(
                query[start:end].transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1),
                scale=scale, mask="causal",
            )
            out[start:end] = attended.transpose(0, 1)
        return out

    def write_kv(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fp8_scales: tuple[float, float] | None = None,
    ) -> None:
        """In place; the kernels skip slot -1 and quantize to fp8 in the same pass."""
        if fp8_scales is not None:
            self.ext.kv_quantize_scatter(key_pages, slot_mapping, key, fp8_scales[0])
            self.ext.kv_quantize_scatter(value_pages, slot_mapping, value, fp8_scales[1])
        else:
            self.ext.paged_cache_update(key_pages, slot_mapping, key)
            self.ext.paged_cache_update(value_pages, slot_mapping, value)
