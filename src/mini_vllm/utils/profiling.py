"""Opt-in instrumentation: named ranges for profiler timelines, and what each step did.

With MINI_VLLM_PROFILE=1 in the environment, or after enable(), span() opens a torch.profiler
record_function range and, on a CUDA machine, an NVTX range, so scheduling, input preparation,
the forward and sampling show up by name in a torch.profiler trace or an Nsight Systems
timeline; and CudaTimer times the forward with CUDA events. Off, span() is a shared no-op
context and nothing is timed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

import torch

__all__ = ["CudaTimer", "StepStats", "enable", "enabled", "span"]

_enabled = os.environ.get("MINI_VLLM_PROFILE") == "1"
_NOTHING = nullcontext()


def enable(on: bool = True) -> None:
    global _enabled
    _enabled = on


def enabled() -> bool:
    return _enabled


@contextmanager
def _range(name: str) -> Iterator[None]:
    nvtx = torch.cuda.is_available()
    with torch.profiler.record_function(name):
        if nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()


def span(name: str) -> AbstractContextManager[None]:
    """A named range around one phase of a step; a no-op unless profiling is enabled."""
    return _range(name) if _enabled else _NOTHING


class CudaTimer:
    """GPU time between start() and stop(). Read elapsed_ms() only once the stream has passed
    stop(), after a sync the step makes anyway, so timing adds none. Inert, returning None,
    unless profiling is enabled and the device is a GPU."""

    def __init__(self, device: torch.device) -> None:
        self._events = None
        if _enabled and device.type == "cuda":
            self._events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

    def start(self) -> None:
        if self._events is not None:
            self._events[0].record()

    def stop(self) -> None:
        if self._events is not None:
            self._events[1].record()

    def elapsed_ms(self) -> float | None:
        return self._events[0].elapsed_time(self._events[1]) if self._events is not None else None


@dataclass(frozen=True)
class StepStats:
    """What one engine step computed: enough to turn its time into bytes moved per second."""

    num_seqs: int             # requests in the forward
    num_tokens: int           # tokens through the forward: 1 per decode, a chunk's length per prefill
    context_tokens: int       # the sum of context lengths: KV entries the attention read
    pure_decode: bool         # every request computed exactly one token, no proposals
    num_preempted: int
    forward_ms: float | None  # GPU time of the target forward; None unless profiling is enabled
