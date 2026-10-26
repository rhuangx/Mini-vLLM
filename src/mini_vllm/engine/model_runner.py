"""Running the model on a scheduled batch: the forward, in the context of the pool it pages
through."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mini_vllm.utils.context import Context, forward_context

if TYPE_CHECKING:
    from mini_vllm.engine.forward_batch import ForwardBatch
    from mini_vllm.engine.kv_cache import PagedKvPool
    from mini_vllm.models.qwen3 import Qwen3Model

__all__ = ["run_batch"]


def run_batch(
    model: Qwen3Model, kv: PagedKvPool, batch: ForwardBatch, rows: torch.Tensor | None = None
) -> torch.Tensor:
    """Logits [rows, V] for one ragged batch, every layer writing and reading kv's pages; rows
    selects the token rows that reach the LM head, None meaning all of them.

    The model and the pool must agree on depth: a self-draft runs the target's first layers
    over a pool of exactly that many."""
    assert len(model.layers) <= kv.num_layers, f"{len(model.layers)} layers over a {kv.num_layers}-layer pool"
    with forward_context(Context(kv=kv, batch=batch)):
        return model(batch.input_ids[None], batch.positions, rows=rows)[0]
