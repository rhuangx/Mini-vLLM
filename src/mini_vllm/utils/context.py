"""The forward context: what attention reads and writes during one forward pass.

Attention layers hold no KV state of their own. Each knows its layer index, and looks up the
storage of the forward it is running in here: a paged pool and the batch's metadata when
serving, a dense cache per layer in the reference loop, or nothing, when a call attends only
over its own tokens. A self-draft shares the target's first layers but caches in a pool of
its own, which works because the pool comes with the forward, not with the layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.engine.forward_batch import ForwardBatch
    from mini_vllm.engine.kv_cache import PagedKvPool
    from mini_vllm.reference.kv_cache import KvFullCache

__all__ = ["Context", "forward_context", "get_context", "reset_context", "set_context"]


@dataclass(frozen=True)
class Context:
    kv: PagedKvPool | None = None                       # serving: the pool this forward pages through
    batch: ForwardBatch | None = None                   # and the ragged batch's slots and tables
    dense_caches: Sequence[KvFullCache] | None = None   # the reference loop: one history per layer


_context = Context()


def get_context() -> Context:
    return _context


def set_context(context: Context) -> None:
    global _context
    _context = context


def reset_context() -> None:
    set_context(Context())


@contextmanager
def forward_context(context: Context) -> Iterator[Context]:
    """Run a forward in `context`, restoring the one before it afterwards, even on an error;
    nested forwards, such as the draft's inside a step, each see their own."""
    previous = get_context()
    set_context(context)
    try:
        yield context
    finally:
        set_context(previous)
