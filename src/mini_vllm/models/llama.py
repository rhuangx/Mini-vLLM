"""Llama 3, which is Qwen3's block without the per-head q/k norm and with Llama 3's RoPE
frequency scaling: both come from the checkpoint's config (ModelConfig.qk_norm, rope_scaling),
so the modules and their names are Qwen3's."""

from __future__ import annotations

from mini_vllm.models.qwen3 import Qwen3ForCausalLM

__all__ = ["LlamaForCausalLM"]


class LlamaForCausalLM(Qwen3ForCausalLM):
    pass
