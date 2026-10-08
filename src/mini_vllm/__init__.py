"""Mini-vLLM: a paged-attention LLM inference engine for NVIDIA GPUs."""

import os

# Before torch: its Triton matmul overrides need Python headers and are not the
# cuBLAS numerics this engine is checked against. Force the flag even if torch
# was imported earlier, then drop any overrides that already registered.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

from .attention import *
from .basics import *
from .batch import *
from .embedding import *
from .engine import *
from .generate import *
from .kv_cache import *
from .layer_norm import *
from .models import *
from .paged_kv_cache import *
from .positional_encoding import *
from .prefix_cache import *
from .quantize import *
from .qwen3 import *
from .sampler import *
from .scheduler import *
from .speculative import *
