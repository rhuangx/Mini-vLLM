"""The paged KV cache's storage: every layer's pages, written through a slot mapping, and the
per-layer view the model attends through."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mini_vllm.layers.attention import paged_attention
from mini_vllm.layers.quantization import quantize_fp8

if TYPE_CHECKING:
    from mini_vllm.engine.forward_batch import ForwardBatch

__all__ = ["PagedKvCache", "PagedKvPool"]


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
        use_cuda: bool = False,
    ) -> None:
        """Scatter key/value [T, H_k, D] into the slots slot_mapping names.

        The write is in place: index_copy_ on the pure path, and the CUDA kernels, which
        quantize to fp8 in the same pass and do not rebind the pool tensors.
        """
        if use_cuda:
            import mini_vllm_ext

            if self.fp8:
                mini_vllm_ext.kv_quantize_scatter(self.keys[layer], slot_mapping, key, self.k_scale)
                mini_vllm_ext.kv_quantize_scatter(self.values[layer], slot_mapping, value, self.v_scale)
            else:
                mini_vllm_ext.paged_cache_update(self.keys[layer], slot_mapping, key)
                mini_vllm_ext.paged_cache_update(self.values[layer], slot_mapping, value)
            return
        if self.fp8:
            key, value = quantize_fp8(key, self.k_scale), quantize_fp8(value, self.v_scale)
        slots = slot_mapping.long()
        self.keys[layer].index_copy_(0, slots, key.to(self.keys[layer].dtype))
        self.values[layer].index_copy_(0, slots, value.to(self.values[layer].dtype))

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

    def caches(self, batch: ForwardBatch) -> list[PagedKvCache]:
        """One view per layer for this batch, to pass to the model as its caches."""
        return [PagedKvCache(self, layer, batch) for layer in range(len(self.keys))]


class PagedKvCache:
    """One layer of the pool, as seen by one ragged batch. The model sees one row of
    T flattened tokens: q is 1 x H_q x T x D, k and v are 1 x H_k x T x D."""

    def __init__(self, pool: PagedKvPool, layer: int, batch: ForwardBatch) -> None:
        self.pool = pool
        self.layer = layer
        self.batch = batch

    def attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_cuda: bool = False
    ) -> torch.Tensor:
        """Write this batch's keys and values into their slots, then attend each sequence
        over its own pages."""
        pool, batch = self.pool, self.batch
        # k, v arrive as 1 x H_k x T x D; the pool wants T x H_k x D, contiguous.
        pool.write(
            self.layer,
            batch.slot_mapping,
            k[0].transpose(0, 1).contiguous(),
            v[0].transpose(0, 1).contiguous(),
            use_cuda,
        )
        key_pages, value_pages = pool.pages(self.layer)
        out = paged_attention(
            q[0].transpose(0, 1),
            key_pages,
            value_pages,
            batch.block_tables,
            batch.cu_seqlens_q,
            batch.context_lens,
            k_scale=pool.k_scale,
            v_scale=pool.v_scale,
            use_cuda=use_cuda,
            host=batch.host,
        )
        return out.transpose(0, 1)[None]
