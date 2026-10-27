"""RMSNorm: the pre-attention, pre-MLP, final and per-head q/k norms."""

from __future__ import annotations

import torch
from torch import nn

from mini_vllm.layers.backends import Backend, get_backend

__all__ = ["RMSNorm", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis, reduced in fp32."""
    input_dtype = x.dtype
    x32 = x.float()
    normalized = x32 * torch.rsqrt(torch.mean(torch.square(x32), dim=-1, keepdim=True) + eps)
    # The cast back happens before the weight multiply, matching Hugging Face exactly.
    return weight * normalized.to(input_dtype)


class RMSNorm(nn.Module):
    """With a residual, adds it first (in the model's dtype, as Hugging Face does) and also
    returns the sum, the next residual: one pass where the backend can fuse the two."""

    def __init__(self, dim: int, eps: float = 1e-6, backend: str | Backend = "torch") -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.backend = get_backend(backend)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.backend.rms_norm(x, self.weight, self.eps)
        return self.backend.add_rms_norm(x, residual, self.weight, self.eps)
