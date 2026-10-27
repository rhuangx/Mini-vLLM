"""The paged KV cache's storage: every layer's pages, written through a slot mapping, and
attention over them for a ragged batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mini_vllm.layers.backends import Backend, get_backend

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import ForwardBatch

__all__ = ["PagedKvPool"]


class PagedKvPool:
    """Every layer's keys and values, allocated once and never grown. Each layer is stored
    flat, num_slots x H_k x D, so a write is one scatter through the slot mapping; the
    paged num_blocks x P x H_k x D view attention reads is a free reshape."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        fp8: bool = False,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # fp8 pages hold e4m3 bits in uint8 at static per-tensor scales: half the bytes of bf16.
        self.fp8 = fp8
        self.k_scale = k_scale
        self.v_scale = v_scale

        shape = (num_blocks * block_size, num_kv_heads, head_dim)
        storage = torch.uint8 if fp8 else dtype
        if device is None:
            device = torch.device("cpu")
        self.keys = [torch.zeros(shape, dtype=storage, device=device) for _ in range(num_layers)]
        self.values = [torch.zeros(shape, dtype=storage, device=device) for _ in range(num_layers)]

    def write(
        self,
        layer: int,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        backend: str | Backend = "torch",
    ) -> None:
        """Scatter key/value [T, H_k, D] into the slots slot_mapping names, in place, so the
        pool tensors are never rebound; fp8 pools quantize on the way in."""
        scales = (self.k_scale, self.v_scale) if self.fp8 else None
        get_backend(backend).write_kv(self.keys[layer], self.values[layer], slot_mapping, key, value, scales)

    def pages(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's keys and values as num_blocks x P x H_k x D."""
        shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        return self.keys[layer].reshape(shape), self.values[layer].reshape(shape)

    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page in every layer: the copy in copy-on-write. Raw values, so an
        fp8 page is copied bit for bit."""
        src = slice(source * self.block_size, (source + 1) * self.block_size)
        dst = slice(destination * self.block_size, (destination + 1) * self.block_size)
        for pool in (*self.keys, *self.values):
            pool[dst] = pool[src]

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def attend(
        self,
        layer: int,
        batch: ForwardBatch,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        backend: str | Backend = "torch",
    ) -> torch.Tensor:
        """One layer's attention for a ragged batch, all [T, H, D]: write the batch's keys and
        values into their slots, then attend each sequence over its own pages."""
        backend = get_backend(backend)
        self.write(layer, batch.slot_mapping, k, v, backend)
        key_pages, value_pages = self.pages(layer)
        return backend.paged_attention(
            q, key_pages, value_pages, batch.block_tables, batch.cu_seqlens_q, batch.context_lens,
            k_scale=self.k_scale, v_scale=self.v_scale, host=batch.host,
            num_short=batch.num_short, max_query_len=batch.max_query_len,
        )
