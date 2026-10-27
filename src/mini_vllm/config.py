"""Configuration: the engine's and the scheduler's settings, and the shape of the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["EngineConfig", "ModelConfig", "SchedulerConfig"]


@dataclass(frozen=True)
class SchedulerConfig:
    max_batched_tokens: int = 2048       # compute budget: token positions per forward pass
    max_sequences: int = 16              # requests in flight at once
    chunk_size: int = 512                # one prefill's share of an iteration
    enable_chunked_prefill: bool = True  # off: a prompt is prefilled in one pass
    prefill_priority: bool = False       # prompts before decodes: the pre-chunking baseline


@dataclass(frozen=True)
class EngineConfig:
    """Everything the engine takes besides the model and tokenizer."""

    backend: str = "torch"          # where operators run: torch, cuda (src/extensions) or triton

    # KV memory.
    num_blocks: int | None = None   # None: what fits in gpu_memory_utilization after a warmup
    block_size: int = 16            # tokens per page
    gpu_memory_utilization: float = 0.9
    fp8_kv_cache: bool = False      # e4m3 pages: half the bytes, so twice the tokens in flight
    enable_prefix_caching: bool = False

    # Scheduling.
    max_batched_tokens: int = 2048
    max_sequences: int = 32
    chunk_size: int = 512
    enable_chunked_prefill: bool = True
    prefill_priority: bool = False
    max_model_len: int = 4096        # the longest context a CUDA graph's block table covers

    # CUDA graphs replay decode steps of up to cuda_graph_max_bs sequences (cuda and triton).
    enforce_eager: bool = False
    cuda_graph_max_bs: int = 256

    # Speculative decoding: k > 0 drafts k tokens per decode step for the target to verify.
    num_speculative_tokens: int = 0
    draft_model: str | None = None        # a separate checkpoint sharing the tokenizer, or
    num_draft_layers: int | None = None   # None: a self-draft of the target's first N layers
    draft_blocks: int | None = None       # None: the target's pool plus room for k per request

    tensor_parallel_size: int = 1  # GPUs the model is split over; from_pretrained only, eager
    seed: int | None = None  # seeds torch's global RNG, CUDA included; greedy is deterministic anyway

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_batched_tokens=self.max_batched_tokens,
            max_sequences=self.max_sequences,
            chunk_size=self.chunk_size,
            enable_chunked_prefill=self.enable_chunked_prefill,
            prefill_priority=self.prefill_priority,
        )


@dataclass(frozen=True)
class ModelConfig:
    """The shape of a Qwen3 or Llama checkpoint, read off a transformers config."""

    architecture: str             # e.g. Qwen3ForCausalLM
    vocab_size: int               # V
    hidden_size: int              # E
    num_hidden_layers: int
    num_attention_heads: int      # H_q
    num_key_value_heads: int      # H_k
    head_dim: int                 # D; note H_q * D != E in Qwen3
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    qk_norm: bool = True          # Qwen3 normalizes each head's q and k; Llama does not
    rope_scaling: dict | None = None  # Llama 3's frequency scaling

    @classmethod
    def from_hf_config(cls, config: Any) -> ModelConfig:
        # transformers 5 keeps rope settings in rope_parameters; older configs used rope_theta
        # and rope_scaling.
        rope = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
        theta = rope.get("rope_theta", getattr(config, "rope_theta", 1_000_000.0))
        kind = rope.get("rope_type", rope.get("type"))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # A config built in code has no architectures list; its class name says the same.
        named = getattr(config, "architectures", None)
        architecture = named[0] if named else type(config).__name__.replace("Config", "ForCausalLM")
        return cls(
            architecture=architecture,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=theta,
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
            qk_norm=architecture.startswith("Qwen3"),
            rope_scaling=rope if kind == "llama3" else None,
        )
