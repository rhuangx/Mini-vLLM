"""The dense KV cache: one layer's keys and values, concatenated along the sequence axis."""

from __future__ import annotations

import torch

from mini_vllm.attention import scaled_dot_product_attention_grouped

__all__ = ["KvFullCache"]


class KvFullCache:
    """Grows by concatenation every step: O(S) copy per append, and the paged cache's oracle."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.offset = 0  # how many positions are cached

    def update_and_fetch(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append key/value [B, H_k, L, D], returning the whole history [B, H_k, S, D]."""
        if self.keys is None:
            self.keys, self.values = key, value
        else:
            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)
        self.offset += key.shape[-2]
        return self.keys, self.values

    def attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_cuda: bool = False
    ) -> torch.Tensor:
        """Append this step's keys and values, then attend causally over the whole history.
        With S > L, the causal mask's S - L offset lines the new tokens up after it."""
        keys, values = self.update_and_fetch(k, v)
        return scaled_dot_product_attention_grouped(q, keys, values, mask="causal", use_cuda=use_cuda)
