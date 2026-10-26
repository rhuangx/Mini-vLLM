"""Engine configuration: everything the engine takes besides the model and the tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EngineConfig", "SchedulerConfig"]


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

    # KV memory.
    num_blocks: int | None = None   # None: kv_fraction of the memory left once the weights are in
    block_size: int = 16            # tokens per page
    kv_fraction: float = 0.5
    fp8_kv_cache: bool = False      # e4m3 pages: half the bytes, so twice the tokens in flight
    enable_prefix_caching: bool = False

    # Scheduling.
    max_batched_tokens: int = 2048
    max_sequences: int = 32
    chunk_size: int = 512
    enable_chunked_prefill: bool = True
    prefill_priority: bool = False

    # Speculative decoding: k > 0 drafts k tokens per decode step for the target to verify.
    num_speculative_tokens: int = 0
    draft_model: str | None = None        # a separate checkpoint sharing the tokenizer, or
    num_draft_layers: int | None = None   # None: a self-draft of the target's first N layers
    draft_blocks: int | None = None       # None: the target's pool plus room for k per request

    seed: int | None = None  # seeds torch's global RNG, CUDA included; greedy is deterministic anyway

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_batched_tokens=self.max_batched_tokens,
            max_sequences=self.max_sequences,
            chunk_size=self.chunk_size,
            enable_chunked_prefill=self.enable_chunked_prefill,
            prefill_priority=self.prefill_priority,
        )
