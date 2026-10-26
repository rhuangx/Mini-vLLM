"""How a request is sampled: temperature, top-k, top-p and the number of completions."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GREEDY", "SamplingParams"]


@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling configuration; temperature=0 is greedy, top_k=0 and top_p=1
    disable truncation."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    n: int = 1  # completions per prompt: one prefill, then n branches sharing its pages

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


GREEDY = SamplingParams(temperature=0.0)
