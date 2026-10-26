"""The CUDA backend: each operator on its src/extensions kernel, and the pure path for the
shapes a kernel does not take."""

from __future__ import annotations

import math

import torch

from mini_vllm.layers.attention import gather_pages
from mini_vllm.layers.backends.torch_backend import TorchBackend

__all__ = ["SHORT_QUERY", "CudaBackend"]

# decode_attention and the paged kernel give every query token its own thread block, so they
# take sequences of at most this many query tokens: decodes and speculative verifies.
SHORT_QUERY = 8


class CudaBackend(TorchBackend):
    name = "cuda"

    def __init__(self) -> None:
        import mini_vllm_ext  # raises here when src/extensions is not built

        self.ext = mini_vllm_ext

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.ext.swiglu(gate, up)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return self.ext.rms_norm(x, weight, eps)

    def rope(
        self, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # The same fp32 tables, gathered in the kernel.
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
        flash_prefill for a bf16, D = 128 prefill; anything else, an additive mask included,
        runs on the pure path."""
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
        host: tuple[list[list[int]], list[int], list[int]] | None = None,
    ) -> torch.Tensor:
        """The sequences computing at most SHORT_QUERY tokens (decodes) go to one paged kernel
        that walks the block tables itself. It gives each query token a thread block, so a
        long chunk would re-read its whole context once per token: those gather their pages
        and take dense_attention instead."""
        if scale is None:
            scale = query.shape[-1] ** -0.5
        starts, lengths = (cu_seqlens_q.tolist(), context_lens.tolist()) if host is None else host[1:]

        short = [i for i in range(len(lengths)) if starts[i + 1] - starts[i] <= SHORT_QUERY]
        if len(short) == len(lengths):
            pages = (key_pages, value_pages, block_tables, cu_seqlens_q, context_lens)
            return self.ext.paged_attention(query, *pages, scale, k_scale, v_scale)

        pieces = {}
        if short:
            # A mixed batch: the decodes as one sub-batch, whose rows are split back out below.
            rows = [row for i in short for row in range(starts[i], starts[i + 1])]
            sub_starts = [0]
            for i in short:
                sub_starts.append(sub_starts[-1] + starts[i + 1] - starts[i])
            chosen = torch.tensor(short, dtype=torch.long, device=block_tables.device)
            attended = self.ext.paged_attention(
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

        outputs = []
        for index, context_len in enumerate(lengths):
            if index in pieces:
                outputs.append(pieces[index])
                continue
            start, end = starts[index], starts[index + 1]
            keys, values = gather_pages(
                key_pages, value_pages, block_tables[index], context_len, k_scale, v_scale, query.dtype
            )
            attended = self.dense_attention(
                query[start:end].transpose(0, 1),
                keys.transpose(0, 1),
                values.transpose(0, 1),
                scale=scale,
                mask="causal",
            )
            outputs.append(attended.transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def write_kv(
        self, pages: torch.Tensor, slot_mapping: torch.Tensor, values: torch.Tensor, scale: float | None
    ) -> None:
        """In place, as the pure path is: the kernels quantize to fp8 in the same pass."""
        if scale is not None:
            self.ext.kv_quantize_scatter(pages, slot_mapping, values, scale)
        else:
            self.ext.paged_cache_update(pages, slot_mapping, values)
