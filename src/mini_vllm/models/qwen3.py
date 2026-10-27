"""Qwen3 as nn.Modules named like the Hugging Face checkpoint, so its tensors load by name; only
q/k/v and gate/up are packed, as packed_modules_mapping says. Serving runs one flat row of
tokens [T]; the forward context (utils/context.py) says where attention reads and writes."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import torch
from torch import nn

from mini_vllm.config import ModelConfig
from mini_vllm.layers.activation import SiluAndMul
from mini_vllm.layers.backends import Backend, get_backend
from mini_vllm.layers.embedding_head import ParallelLMHead, VocabParallelEmbedding
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from mini_vllm.layers.rotary_embedding import RotaryEmbedding
from mini_vllm.utils.context import get_context

__all__ = ["Qwen3Attention", "Qwen3DecoderLayer", "Qwen3ForCausalLM", "Qwen3MLP", "Qwen3Model"]


class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding,
                 backend: str | Backend = "torch") -> None:
        super().__init__()
        self.layer_idx = layer_idx  # which layer of the forward's KV storage is this layer's
        self.backend = get_backend(backend)
        d = self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size, d, config.num_attention_heads,
                                          config.num_key_value_heads)
        self.num_heads, self.num_kv_heads = self.qkv_proj.num_heads, self.qkv_proj.num_kv_heads
        self.o_proj = RowParallelLinear(config.num_attention_heads * d, config.hidden_size)
        if config.qk_norm:
            self.q_norm = RMSNorm(d, config.rms_norm_eps, backend)
            self.k_norm = RMSNorm(d, config.rms_norm_eps, backend)
        else:
            self.q_norm = self.k_norm = None
        self.rotary_emb = rotary_emb

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        d, heads, kv_heads = self.head_dim, self.num_heads, self.num_kv_heads
        q, k, v = self.qkv_proj(hidden).split([heads * d, kv_heads * d, kv_heads * d], -1)
        q, k, v = q.view(-1, heads, d), k.view(-1, kv_heads, d), v.view(-1, kv_heads, d)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        # The forward's pool takes this step's k and v and attends over each ragged
        # sequence's own pages; without one, the row is a single sequence seeing itself.
        context = get_context()
        if context.kv is not None:
            out = context.kv.attend(self.layer_idx, context.batch, q, k, v, self.backend)
        else:
            rows = [x.transpose(0, 1)[None] for x in (q, k, v)]
            out = self.backend.dense_attention(*rows, mask="causal")[0].transpose(0, 1)
        return self.o_proj(out.reshape(-1, self.num_heads * d))


class Qwen3MLP(nn.Module):
    """down(silu(gate(x)) * up(x)), gate and up in one GEMM."""

    def __init__(self, config: ModelConfig, backend: str | Backend = "torch") -> None:
        super().__init__()
        size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [size, size])
        self.down_proj = RowParallelLinear(size, config.hidden_size)
        self.act_fn = SiluAndMul(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm residual block; each norm also adds the running residual, in one pass."""

    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding,
                 backend: str | Backend = "torch") -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend)
        self.self_attn = Qwen3Attention(config, layer_idx, rotary_emb, backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend)
        self.mlp = Qwen3MLP(config, backend)

    def forward(
        self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual, hidden = hidden, self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, residual = self.post_attention_layernorm(self.self_attn(positions, hidden), residual)
        return self.mlp(hidden), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig, backend: str | Backend = "torch") -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # One set of rotary tables, shared by every layer.
        rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta,
                                     config.rope_scaling, backend)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, index, rotary_emb, backend) for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    """Token ids [T] at positions [T] -> hidden states [T, E]; compute_logits takes the rows
    that need them to logits [rows, V]."""

    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: ModelConfig, backend: str | Backend = "torch") -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, backend)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor | None:
        """Logits for the selected rows (all when None): one per sequence when serving, rather
        than a V-wide row for every prompt token in a chunk. None on tensor-parallel ranks > 0."""
        if rows is not None:
            hidden = hidden[rows.long()]
        return self.lm_head(hidden)

    def share_prefix(self, num_layers: int) -> Qwen3ForCausalLM:
        """A draft made of this model's first num_layers layers and its head, sharing every
        weight. A fresh shell, not a copy: copy.copy would share the module dict and truncate
        this model too."""
        draft = type(self).__new__(type(self))
        nn.Module.__init__(draft)
        draft.config = replace(self.config, num_hidden_layers=num_layers)
        inner = Qwen3Model.__new__(Qwen3Model)
        nn.Module.__init__(inner)
        inner.embed_tokens, inner.norm = self.model.embed_tokens, self.model.norm
        inner.layers = nn.ModuleList(self.model.layers[:num_layers])
        draft.model, draft.lm_head = inner, self.lm_head
        return draft
