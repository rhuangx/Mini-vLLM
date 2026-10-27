"""Activations: softmax, silu, and the SwiGLU that gates the MLP."""

from __future__ import annotations

import torch
from torch import nn

from mini_vllm.layers.backends import Backend, get_backend

__all__ = ["SiluAndMul", "silu", "silu_and_mul", "softmax"]


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax along axis, computed in fp32 and returned in the input dtype."""
    x32 = x.float()
    # Subtract the row max: exp overflows around 88 in fp32 and the factor cancels.
    x32 = x32 - torch.amax(x32, dim=axis, keepdim=True)
    exp = torch.exp(x32)
    return (exp / torch.sum(exp, dim=axis, keepdim=True)).to(x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, with gate and up the two halves of x [..., 2I]: the merged gate_up
    projection's output."""
    gate, up = x.chunk(2, dim=-1)
    return silu(gate) * up


class SiluAndMul(nn.Module):
    def __init__(self, backend: str | Backend = "torch") -> None:
        super().__init__()
        self.backend = get_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.silu_and_mul(x)
