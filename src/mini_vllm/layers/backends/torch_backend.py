"""The pure-PyTorch backend: the layers' own reference math, on any device, CPU included."""

from __future__ import annotations

import torch

from mini_vllm.layers import activation, attention, layernorm, rotary_embedding
from mini_vllm.layers.quantization import quantize_fp8

__all__ = ["TorchBackend"]


class TorchBackend:
    name = "torch"

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return activation.swiglu(gate, up)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return layernorm.rms_norm(x, weight, eps)

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
        host: tuple[list[list[int]], list[int], list[int]] | None = None,
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
            key, value = quantize_fp8(key, fp8_scales[0]), quantize_fp8(value, fp8_scales[1])
        slots = slot_mapping.long()  # once for both: index_copy_ wants int64
        key_pages.index_copy_(0, slots, key.to(key_pages.dtype))
        value_pages.index_copy_(0, slots, value.to(value_pages.dtype))
