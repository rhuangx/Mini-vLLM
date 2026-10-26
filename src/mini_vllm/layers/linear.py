"""The linear layer: every projection in the model, and the untied LM head."""

from __future__ import annotations

import torch

__all__ = ["linear"]


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ w.T (+ bias), with w stored O x I as the checkpoint keeps it."""
    out = x @ w.T
    if bias is not None:
        out = out + bias
    return out
