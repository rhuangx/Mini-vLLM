"""RoPE: precomputed rotary tables, applied at explicit absolute positions."""

from __future__ import annotations

import torch

from mini_vllm.layers.backends import Backend, get_backend

__all__ = ["RoPE", "apply_rope", "rotate_half"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], pairing element i with element i + D/2 as Qwen3 was trained."""
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


class RoPE:
    """Rotate x [B, L, H, D] by the angles at positions [B, L] or [L]."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 1_000_000.0,
        backend: str | Backend = "torch",
        device: torch.device | None = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency i decays as theta^(-2i/D): early pairs are local, late ones long-range.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inverse_frequencies)  # max_seq_len x D/2

        # Duplicated rather than interleaved, matching `rotate_half`.
        angles = torch.cat([angles, angles], dim=-1)  # max_seq_len x D

        # Tables stay fp32: a bf16 cosine near a zero crossing would shift tokens.
        self.cos = torch.cos(angles)
        self.sin = torch.sin(angles)

    def _tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos.device != device:
            self.cos = self.cos.to(device)
            self.sin = self.sin.to(device)
        return self.cos, self.sin

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos_table, sin_table = self._tables(x.device)
        return self.backend.rope(x, positions, cos_table, sin_table)
