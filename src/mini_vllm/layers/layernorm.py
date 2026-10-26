"""RMSNorm: the pre-attention, pre-MLP, final and per-head q/k norms."""

from __future__ import annotations

import torch

__all__ = ["RMSNorm"]


class RMSNorm:
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis, reduced in fp32."""

    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6, use_cuda: bool = False) -> None:
        self.dim = dim
        self.weight = weight
        self.eps = eps
        self.use_cuda = use_cuda

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda:
            import mini_vllm_ext

            return mini_vllm_ext.rms_norm(x, self.weight, self.eps)

        input_dtype = x.dtype

        x32 = x.float()
        mean_square = torch.mean(torch.square(x32), dim=-1, keepdim=True)
        normalized = x32 * torch.rsqrt(mean_square + self.eps)

        # The cast back happens before the weight multiply, matching HuggingFace exactly.
        return self.weight * normalized.to(input_dtype)
