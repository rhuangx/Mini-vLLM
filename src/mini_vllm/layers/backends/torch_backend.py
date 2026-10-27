"""The pure-PyTorch backend: the layers' own reference math, on any device, CPU included."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mini_vllm.layers import activation, attention, layernorm, rotary_embedding

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import HostLists

__all__ = ["TorchBackend"]


class TorchBackend:
    name = "torch"
    supports_graphs = False  # the paged path loops over sequences in Python

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        return activation.silu_and_mul(x)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return layernorm.rms_norm(x, weight, eps)

    def add_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summed = x + residual
        return self.rms_norm(summed, weight, eps), summed

    def rope(
        self, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return rotary_embedding.apply_rope(x, positions, cos, sin)

    def dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        return attention.scaled_dot_product_attention_grouped(query, key, value, scale, mask)

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
        return attention.paged_attention(
            query, key_pages, value_pages, block_tables, cu_seqlens_q, context_lens,
            scale, k_scale, v_scale, host,
        )

    def write_kv(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fp8_scales: tuple[float, float] | None = None,
    ) -> None:
        if fp8_scales is not None:
            key = attention.quantize_fp8(key, fp8_scales[0])
            value = attention.quantize_fp8(value, fp8_scales[1])
        slots = slot_mapping.long()  # once for both; never -1 here, as only captured batches pad
        key_pages.index_copy_(0, slots, key.to(key_pages.dtype))
        value_pages.index_copy_(0, slots, value.to(value_pages.dtype))
