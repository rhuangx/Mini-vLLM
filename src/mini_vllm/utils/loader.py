"""Loading: transformers reads the checkpoint, then its weights are moved onto our own Qwen3 classes."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.config import ModelConfig
from mini_vllm.layers.embedding_head import Embedding
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.layers.rotary_embedding import RoPE
from mini_vllm.models.qwen3 import Qwen3MLP, Qwen3Model, Qwen3MultiHeadAttention, Qwen3TransformerBlock
from mini_vllm.utils.tokenizer import Tokenizer, eos_token_ids

__all__ = ["DEFAULT_MODEL", "from_hf", "load"]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def _frozen(weight: torch.Tensor) -> torch.Tensor:
    """Share the checkpoint storage without tracking a backward graph through it."""
    return weight.detach()


def from_hf(hf_model: Any, use_cuda: bool = False) -> Qwen3Model:
    """Read every weight off a transformers Qwen3ForCausalLM; nothing of it runs after this.
    use_cuda routes every operator with a kernel through src/extensions/. The attribute paths
    match Qwen3ForCausalLM, and a tied checkpoint leaves lm_head as None. The weights are the
    module's own tensors, shared by reference."""
    config = ModelConfig.from_hf_config(hf_model.config)
    device = hf_model.model.embed_tokens.weight.device

    def norm(dim: int, weight: torch.Tensor) -> RMSNorm:
        return RMSNorm(dim, weight, config.rms_norm_eps, use_cuda)

    # One set of rotary tables, shared by every layer, on the same device as the weights.
    rope = RoPE(config.head_dim, config.max_position_embeddings, config.rope_theta, use_cuda, device=device)

    layers = []
    for layer in hf_model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        attention = Qwen3MultiHeadAttention(
            config,
            _frozen(attn.q_proj.weight),
            _frozen(attn.k_proj.weight),
            _frozen(attn.v_proj.weight),
            _frozen(attn.o_proj.weight),
            norm(config.head_dim, _frozen(attn.q_norm.weight)),
            norm(config.head_dim, _frozen(attn.k_norm.weight)),
            rope,
            use_cuda,
        )
        layers.append(
            Qwen3TransformerBlock(
                attention,
                Qwen3MLP(
                    _frozen(mlp.gate_proj.weight),
                    _frozen(mlp.up_proj.weight),
                    _frozen(mlp.down_proj.weight),
                    use_cuda,
                ),
                norm(config.hidden_size, _frozen(layer.input_layernorm.weight)),
                norm(config.hidden_size, _frozen(layer.post_attention_layernorm.weight)),
            )
        )

    embed = _frozen(hf_model.model.embed_tokens.weight)
    lm_head = None if config.tie_word_embeddings else _frozen(hf_model.lm_head.weight)
    return Qwen3Model(
        config,
        Embedding(config.vocab_size, config.hidden_size, embed),
        layers,
        norm(config.hidden_size, _frozen(hf_model.model.norm.weight)),
        lm_head,
    )


def load(model: str = DEFAULT_MODEL, use_cuda: bool = False) -> tuple[Qwen3Model, Tokenizer]:
    """Load a BF16 Qwen3 checkpoint from the Hugging Face hub: (model, tokenizer)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device)
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer = Tokenizer(hf_tokenizer, eos_token_ids(hf_model, hf_tokenizer))
    qwen3 = from_hf(hf_model, use_cuda)
    # The transplanted weights keep the storage alive; the module itself is not needed.
    del hf_model
    return qwen3, tokenizer
