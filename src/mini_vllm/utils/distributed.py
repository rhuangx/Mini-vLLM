"""Tensor parallelism: this process's rank among the GPUs a model is split over, and the two
collectives the layers need. Without torch.distributed initialized, rank 0 of 1 and no-ops."""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["all_reduce", "gather_to_first", "tp_rank", "tp_size"]


def tp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def tp_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum x over every rank, in place: a row-parallel layer's partial outputs."""
    if tp_size() > 1:
        dist.all_reduce(x)
    return x


def gather_to_first(x: torch.Tensor) -> torch.Tensor | None:
    """Concatenate every rank's x along the last axis on rank 0, which alone samples; other
    ranks get None. The LM head's vocabulary shards."""
    if tp_size() == 1:
        return x
    parts = [torch.empty_like(x) for _ in range(tp_size())] if tp_rank() == 0 else None
    dist.gather(x, parts, dst=0)
    return torch.cat(parts, dim=-1) if parts is not None else None
