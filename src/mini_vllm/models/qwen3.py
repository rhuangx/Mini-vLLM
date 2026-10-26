"""Qwen3, assembled from the operators: attention, MLP, transformer block, model."""

from __future__ import annotations

import torch

from mini_vllm.config import ModelConfig
from mini_vllm.engine.kv_cache import PagedKvCache
from mini_vllm.layers.backends import Backend, get_backend
from mini_vllm.layers.embedding_head import Embedding
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.layers.linear import linear
from mini_vllm.layers.rotary_embedding import RoPE
from mini_vllm.reference.kv_cache import KvFullCache

# What a layer attends through: a dense history, or one layer's view of a paged batch.
Cache = KvFullCache | PagedKvCache

__all__ = ["Qwen3MLP", "Qwen3Model", "Qwen3MultiHeadAttention", "Qwen3TransformerBlock"]


class Qwen3MultiHeadAttention:
    """Grouped-query attention with a per-head RMSNorm on q and k before RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        rope: RoPE,
        backend: str | Backend = "torch",
    ) -> None:
        self.backend = get_backend(backend)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_norm, self.k_norm = q_norm, k_norm
        self.rope = rope

    def __call__(self, x: torch.Tensor, positions: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        batch, length, _ = x.shape

        q = linear(x, self.wq).reshape(batch, length, self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(batch, length, self.num_kv_heads, self.head_dim)

        # RoPE rotates B x L x H x D; attention wants B x H x L x D.
        q = self.rope(self.q_norm(q), positions).transpose(1, 2)
        k = self.rope(self.k_norm(k), positions).transpose(1, 2)
        v = v.transpose(1, 2)

        # The cache writes this step's k and v and attends over everything it holds: a dense
        # history, or each ragged sequence's own pages. Without one, the tokens see only each other.
        if cache is None:
            out = self.backend.dense_attention(q, k, v, mask="causal")
        else:
            out = cache.attend(q, k, v, self.backend)
        return linear(out.transpose(1, 2).reshape(batch, length, -1), self.wo)


class Qwen3MLP:
    """down(silu(gate(x)) * up(x))."""

    def __init__(
        self, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, backend: str | Backend = "torch"
    ) -> None:
        self.w_gate, self.w_up, self.w_down = w_gate, w_up, w_down
        self.backend = get_backend(backend)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = linear(x, self.w_gate), linear(x, self.w_up)
        return linear(self.backend.swiglu(gate, up), self.w_down)


class Qwen3TransformerBlock:
    """Pre-norm residual block: x + attention(norm(x)), then h + mlp(norm(h))."""

    def __init__(
        self,
        attention: Qwen3MultiHeadAttention,
        mlp: Qwen3MLP,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
        self.attention = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def __call__(self, x: torch.Tensor, positions: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        h = x + self.attention(self.input_layernorm(x), positions, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3Model:
    """Token ids [B, L] at positions [L] or [B, L] -> logits [B, L, V].

    rows selects which token rows get logits; None means all of them.

    With no caches every call attends only over its own tokens: the full-recompute oracle.
    With a KvFullCache per layer, each call appends to one dense history per row. With the
    PagedKvCache views of a ForwardBatch, B = 1 and L is the batch's T flattened tokens.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding: Embedding,
        layers: list[Qwen3TransformerBlock],
        norm: RMSNorm,
        lm_head: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head  # None when tied to the embedding table

    def __call__(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        caches: list[Cache] | None = None,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embedding(inputs)
        for index, layer in enumerate(self.layers):
            h = layer(h, positions, None if caches is None else caches[index])

        # Only these token rows reach the LM head: one per sequence when serving, rather
        # than a V-wide row for every prompt token in a chunk.
        if rows is not None:
            h = h[:, rows.long()]
        h = self.norm(h)

        if self.lm_head is None:
            return self.embedding.as_linear(h)
        return linear(h, self.lm_head)
