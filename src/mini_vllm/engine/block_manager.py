"""Page bookkeeping: refcounted block ids, per-request block tables, prefix-cache reuse, and
copy-on-write. Which pages a request owns; the pages' contents live in engine/kv_cache.py."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from mini_vllm.engine.prefix_cache import PrefixCache

if TYPE_CHECKING:
    from mini_vllm.engine.kv_cache import PagedKvPool
    from mini_vllm.engine.sequence import Request

__all__ = ["BlockManager", "BlockPool", "BlockTable"]


class BlockPool:
    """A fixed set of physical block ids, handed out and refcounted."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.ref_counts = [0] * num_blocks
        # FIFO, not LIFO, so a use-after-free reads another sequence's page, not its own.
        self.free: deque[int] = deque(range(num_blocks))

        # Prefix caching: a cached block stays matchable on the free list until reused.
        self.cached = [False] * num_blocks
        self.on_evict: Callable[[int], None] | None = None

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self) -> int:
        """Take the oldest free block at reference count 1."""
        block_id = self.free.popleft()
        # Repurposing a cached page: unlink it first, so no later match can return it.
        if self.cached[block_id]:
            self.cached[block_id] = False
            self.on_evict(block_id)
        self.ref_counts[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Add a holder: a fork, or a prefix-cache hit on a block still in use."""
        self.ref_counts[block_id] += 1

    def decref(self, block_id: int) -> bool:
        """Drop a holder, returning True if that freed the block."""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id]:
            return False
        self.free.append(block_id)
        return True

    def acquire_cached(self, block_id: int) -> None:
        """Hold a matched block, from wherever it sits: in use, or free but still cached."""
        if self.ref_counts[block_id] == 0:
            self.free.remove(block_id)
        self.ref_counts[block_id] += 1


class BlockTable:
    """The physical block ids backing one sequence, in logical order, and how many of
    their slots hold tokens."""

    def __init__(self, block_size: int, block_ids: list[int] | None = None, num_tokens: int = 0) -> None:
        self.block_size = block_size
        self.block_ids = [] if block_ids is None else list(block_ids)
        self.num_tokens = num_tokens

    @property
    def num_slots(self) -> int:
        """Token capacity, not a range of physical slot numbers."""
        return len(self.block_ids) * self.block_size

    @property
    def num_empty_slots(self) -> int:
        """Room left in the last block."""
        return self.num_slots - self.num_tokens

    def blocks_needed_for(self, num_new_tokens: int) -> int:
        """How many fresh blocks appending num_new_tokens would take; often zero."""
        deficit = num_new_tokens - self.num_empty_slots
        return max(0, -(-deficit // self.block_size))

    def append_block(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    def append_tokens(self, count: int) -> None:
        self.num_tokens += count

    def replace_block(self, index: int, block_id: int) -> int:
        """Repoint one entry, returning the id it displaced for the caller to decref."""
        displaced = self.block_ids[index]
        self.block_ids[index] = block_id
        return displaced

    def copy(self) -> BlockTable:
        """An independent table over the same blocks; the caller increfs them."""
        return BlockTable(self.block_size, self.block_ids, self.num_tokens)

    def slots(self, start: int, stop: int) -> list[int]:
        """The flat pool slot of each position in [start, stop): block_id * P + offset."""
        size = self.block_size
        return [self.block_ids[p // size] * size + p % size for p in range(start, stop)]


class BlockManager:
    """Capacity, growth, sharing and release: the pool made usable by a scheduler.

    It owns the block pool and the KV pool together, and reaches each request's block
    table through the request, because copy-on-write changes a refcount, a table entry
    and a page at once.
    """

    def __init__(self, kv: PagedKvPool, enable_prefix_caching: bool = False) -> None:
        self.kv = kv
        self.block_size = kv.block_size
        self.pool = BlockPool(kv.num_blocks)
        self.cache = None
        if enable_prefix_caching:
            self.cache = PrefixCache(self.block_size)
            self.pool.on_evict = self.cache.evict

    def blocks_needed(self, request: Request, num_tokens: int) -> int:
        """Fresh blocks num_tokens more would take, counting a copy-on-write copy."""
        table = request.block_table
        if table is None:
            return -(-num_tokens // self.block_size)
        return table.blocks_needed_for(num_tokens) + self._needs_copy(table)

    def allocate(self, request: Request, num_tokens: int) -> None:
        """Reserve slots for num_tokens more tokens, creating or extending the table."""
        if request.block_table is None:
            request.block_table = BlockTable(self.block_size)
        table = request.block_table

        self._resolve_copy_on_write(table)
        for _ in range(table.blocks_needed_for(num_tokens)):
            table.append_block(self.pool.allocate())
        table.append_tokens(num_tokens)

    def slots(self, request: Request, num_tokens: int) -> list[int]:
        """Where the last num_tokens reserved tokens are written."""
        table = request.block_table
        return table.slots(table.num_tokens - num_tokens, table.num_tokens)

    def apply_prefix_cache(self, request: Request) -> int:
        """Reuse whatever of a fresh request's tokens the cache holds, returning how many."""
        if self.cache is None or request.block_table is not None:
            return 0

        matched = self.cache.match(request.token_ids)
        # A full match would leave nothing to forward, and so no logits to sample from.
        while matched and len(matched) * self.block_size >= len(request):
            matched.pop()
        if not matched:
            return 0

        for block_id in matched:
            self.pool.acquire_cached(block_id)
        reused = len(matched) * self.block_size
        request.block_table = BlockTable(self.block_size, matched, reused)
        request.num_computed_tokens = reused
        return reused

    def trim(self, request: Request, num_tokens: int) -> int:
        """Give back the last num_tokens slots, a rejected speculative tail, returning how
        many blocks that freed."""
        table = request.block_table
        table.num_tokens -= num_tokens
        blocks_used = -(-table.num_tokens // self.block_size)
        released = 0
        while len(table.block_ids) > blocks_used:
            released += self.pool.decref(table.block_ids.pop())
        return released

    def fork(self, parent: Request, child: Request) -> None:
        """Point child at every one of parent's blocks, copying nothing."""
        for block_id in parent.block_table.block_ids:
            self.pool.incref(block_id)
        child.block_table = parent.block_table.copy()

    def free(self, request: Request) -> int:
        """Drop a holder on every block the request owns, returning how many were freed."""
        table = request.block_table
        if table is None:
            return 0

        # Cache the full blocks first: they stay matchable while they sit on the free list.
        if self.cache is not None:
            num_full = table.num_tokens // self.block_size
            tokens = request.token_ids[: num_full * self.block_size]
            for block_id in self.cache.insert(tokens, table.block_ids[:num_full]):
                self.pool.cached[block_id] = True

        request.block_table = None
        return sum(self.pool.decref(block_id) for block_id in table.block_ids)

    def _needs_copy(self, table: BlockTable) -> bool:
        """Whether the next write into this table would land on a shared page."""
        if not table.block_ids or table.num_empty_slots == 0:
            return False
        return self.pool.ref_counts[table.block_ids[-1]] > 1

    def _resolve_copy_on_write(self, table: BlockTable) -> None:
        """Give the table a private copy of its last page if it is sharing one."""
        if not self._needs_copy(table):
            return
        # Allocate, copy, repoint, then decref: the other order could hand out the source.
        fresh = self.pool.allocate()
        self.kv.copy_block(table.block_ids[-1], fresh)
        self.pool.decref(table.replace_block(len(table.block_ids) - 1, fresh))
