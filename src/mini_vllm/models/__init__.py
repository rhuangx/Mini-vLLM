"""Model architectures assembled from the layers, by the name a checkpoint's config gives."""

from mini_vllm.models.llama import LlamaForCausalLM
from mini_vllm.models.qwen3 import Qwen3ForCausalLM

MODELS = {"Qwen3ForCausalLM": Qwen3ForCausalLM, "LlamaForCausalLM": LlamaForCausalLM}

__all__ = ["MODELS"]
