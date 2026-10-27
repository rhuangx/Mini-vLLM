"""The CUDA kernels, built by src/extensions/build.py. Every op here has a pure-torch
counterpart in mini_vllm that is its test oracle. The fake (meta) versions say what each op
returns without running it, which torch.compile needs to trace through them."""

from pathlib import Path

import torch

torch.ops.load_library(str(Path(__file__).parent / "_ext.so"))

ops = torch.ops.mini_vllm_ext
rms_norm = ops.rms_norm
silu_and_mul = ops.silu_and_mul
rope = ops.rope
paged_cache_update = ops.paged_cache_update
paged_attention = ops.paged_attention
decode_attention = ops.decode_attention
flash_prefill = ops.flash_prefill
kv_quantize_scatter = ops.kv_quantize_scatter


@torch.library.register_fake("mini_vllm_ext::rms_norm")
def _(x, weight, eps):
    return torch.empty_like(x)


@torch.library.register_fake("mini_vllm_ext::silu_and_mul")
def _(x):
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.register_fake("mini_vllm_ext::rope")
def _(x, positions, cos, sin):
    return torch.empty_like(x)


@torch.library.register_fake("mini_vllm_ext::paged_attention")
def _(q, key_pages, value_pages, block_tables, cu_seqlens_q, context_lens, scale, k_scale, v_scale):
    return torch.empty_like(q)


@torch.library.register_fake("mini_vllm_ext::decode_attention")
def _(q, k, v, scale, causal):
    return torch.empty_like(q)


@torch.library.register_fake("mini_vllm_ext::flash_prefill")
def _(q, k, v, scale, causal):
    return torch.empty_like(q)


@torch.library.register_fake("mini_vllm_ext::paged_cache_update")
def _(pool, slot_mapping, values):
    return None


@torch.library.register_fake("mini_vllm_ext::kv_quantize_scatter")
def _(pool, slot_mapping, values, scale):
    return None
