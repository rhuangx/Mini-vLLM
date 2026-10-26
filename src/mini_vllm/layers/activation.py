"""Activations: softmax, silu, and the SwiGLU that gates Qwen3's MLP."""

from __future__ import annotations

import torch

__all__ = ["silu", "softmax", "swiglu"]


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax along axis, computed in fp32 and returned in the input dtype."""
    x32 = x.float()
    # Subtract the row max: exp overflows around 88 in fp32 and the factor cancels.
    x32 = x32 - torch.amax(x32, dim=axis, keepdim=True)
    exp = torch.exp(x32)
    return (exp / torch.sum(exp, dim=axis, keepdim=True)).to(x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """x * sigmoid(x), the activation inside Qwen3's SwiGLU MLP."""
    return x * torch.sigmoid(x)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up: the MLP's two up-projections merged before the down-projection."""
    return silu(gate) * up
