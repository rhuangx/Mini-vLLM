"""Mini-vLLM: a paged-attention LLM inference engine for NVIDIA GPUs."""

import os

# Before torch: its Triton matmul overrides need Python headers and are not the
# cuBLAS numerics this engine is checked against. Force the flag even if torch
# was imported earlier, then drop any overrides that already registered.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

from .config import *
from .engine.block_manager import *
from .engine.forward_batch import *
from .engine.kv_cache import *
from .engine.llm_engine import *
from .engine.prefix_cache import *
from .engine.scheduler import *
from .engine.sequence import *
from .engine.speculative import *
from .layers.activation import *
from .layers.attention import *
from .layers.embedding_head import *
from .layers.layernorm import *
from .layers.linear import *
from .layers.quantization import *
from .layers.rotary_embedding import *
from .layers.sampler import *
from .models.qwen3 import *
from .reference.generate import *
from .reference.kv_cache import *
from .sampling_parameters import *
from .utils.loader import *
from .utils.tokenizer import *
