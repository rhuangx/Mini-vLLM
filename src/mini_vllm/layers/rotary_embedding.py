"""RoPE: precomputed rotary tables, applied at explicit absolute positions, with Llama 3's
frequency scaling when the checkpoint asks for it."""

from __future__ import annotations

import math

import torch
from torch import nn

from mini_vllm.layers.backends import Backend, get_backend

__all__ = ["RotaryEmbedding", "apply_rope", "rotate_half"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], pairing element i with element i + D/2 as Qwen3 and Llama train."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor
) -> torch.Tensor:
    """Rotate x [B, L, H, D] by the fp32 table rows at positions [L] or [B, L]."""
    # The head axis is inserted so one table row applies to every head of its token.
    cos = cos_table[positions.long()].unsqueeze(-2)
    sin = sin_table[positions.long()].unsqueeze(-2)
    x32 = x.float()
    return (x32 * cos + rotate_half(x32) * sin).to(x.dtype)


def _llama3_frequencies(inverse: torch.Tensor, scaling: dict) -> torch.Tensor:
    """Llama 3's long-context fix: low frequencies slowed by `factor`, high ones kept, and a
    smooth blend between."""
    factor, low, high = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
    old = scaling["original_max_position_embeddings"]
    wavelength = 2 * math.pi / inverse
    scaled = torch.where(wavelength > old / low, inverse / factor, inverse)
    smooth = (old / wavelength - low) / (high - low)
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    medium = (wavelength >= old / high) & (wavelength <= old / low)
    return torch.where(medium, blended, scaled)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float = 1_000_000.0,
        scaling: dict | None = None,
        backend: str | Backend = "torch",
    ) -> None:
        super().__init__()
        self.backend = get_backend(backend)
        # Frequency i decays as base^(-2i/D): early pairs are local, late ones long-range.
        inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if scaling is not None:
            inverse = _llama3_frequencies(inverse, scaling)
        angles = torch.outer(torch.arange(max_position, dtype=torch.float32), inverse)
        # Duplicated rather than interleaved, matching rotate_half; fp32, as a bf16 cosine near
        # a zero crossing would shift tokens.
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q [T, H_q, D] and k [T, H_k, D], each token rotated by its own position."""
        rope, cos, sin = self.backend.rope, self.cos, self.sin
        return rope(q[None], positions, cos, sin)[0], rope(k[None], positions, cos, sin)[0]
