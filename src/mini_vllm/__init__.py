"""Mini-vLLM: a paged-attention LLM inference engine for NVIDIA GPUs.

The serving API is here; everything else is imported from its module, e.g.
mini_vllm.engine.block_manager or mini_vllm.layers.attention.
"""

import os

# Before torch: its Triton matmul overrides need Python headers and are not the
# cuBLAS numerics this engine is checked against.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

from mini_vllm.config import EngineConfig, ModelConfig, SchedulerConfig
from mini_vllm.engine.llm_engine import LLM, Completion, StreamUpdate
from mini_vllm.engine.sequence import Request, RequestStatus
from mini_vllm.sampling_parameters import GREEDY, SamplingParams
from mini_vllm.utils.loader import DEFAULT_MODEL, Tokenizer, from_hf, load

__all__ = [
    "DEFAULT_MODEL",
    "GREEDY",
    "LLM",
    "Completion",
    "EngineConfig",
    "ModelConfig",
    "Request",
    "RequestStatus",
    "SamplingParams",
    "SchedulerConfig",
    "StreamUpdate",
    "Tokenizer",
    "from_hf",
    "load",
]
