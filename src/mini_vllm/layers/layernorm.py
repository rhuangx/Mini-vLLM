"""RMSNorm: the pre-attention, pre-MLP, final and per-head q/k norms."""

from __future__ import annotations

import torch

from mini_vllm.layers.backends import Backend, get_backend

__all__ = ["RMSNorm", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis, reduced in fp32."""
    input_dtype = x.dtype

    x32 = x.float()
    mean_square = torch.mean(torch.square(x32), dim=-1, keepdim=True)
    normalized = x32 * torch.rsqrt(mean_square + eps)

    # The cast back happens before the weight multiply, matching HuggingFace exactly.
    return weight * normalized.to(input_dtype)


class RMSNorm:
    """rms_norm with its weight and eps, on a backend."""

    def __init__(
        self, dim: int, weight: torch.Tensor, eps: float = 1e-6, backend: str | Backend = "torch"
    ) -> None:
        self.dim = dim
        self.weight = weight
        self.eps = eps
        self.backend = get_backend(backend)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.rms_norm(x, self.weight, self.eps)
