"""The CUDA kernels, built by src/extensions/build.py. Every op here has a pure-torch
counterpart in mini_vllm that is its test oracle."""

from pathlib import Path

import torch

torch.ops.load_library(str(Path(__file__).parent / "_ext.so"))

rms_norm = torch.ops.mini_vllm_ext.rms_norm
swiglu = torch.ops.mini_vllm_ext.swiglu
rope = torch.ops.mini_vllm_ext.rope
paged_cache_update = torch.ops.mini_vllm_ext.paged_cache_update
paged_attention = torch.ops.mini_vllm_ext.paged_attention
decode_attention = torch.ops.mini_vllm_ext.decode_attention
flash_prefill = torch.ops.mini_vllm_ext.flash_prefill
kv_quantize_scatter = torch.ops.mini_vllm_ext.kv_quantize_scatter
