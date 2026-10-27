"""Running the model: the ragged batch it takes, the KV pool sized to the GPU, decode steps
replayed from CUDA graphs, and, with tensor parallelism, the worker processes that run their
shard of every forward."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from mini_vllm.engine.kv_cache import PagedKvPool
from mini_vllm.layers.backends import SHORT_QUERY, get_backend
from mini_vllm.utils.context import Context, forward_context
from mini_vllm.utils.distributed import tp_size

if TYPE_CHECKING:
    from mini_vllm.config import EngineConfig
    from mini_vllm.engine.block_manager import BlockManager
    from mini_vllm.engine.sequence import Request

__all__ = [
    "PADDING_BLOCK",
    "ForwardBatch",
    "HostLists",
    "ModelRunner",
    "decode_first",
    "init_distributed",
    "run_batch",
]

# What an unused block-table entry holds: an impossible id, not a silent alias for block 0.
PADDING_BLOCK = -1


class HostLists(NamedTuple):
    """A batch's metadata as Python lists: what the tensors were built from, read instead of
    syncing them back, and what tensor-parallel workers receive."""

    input_ids: list[int]
    positions: list[int]
    offsets: list[int]           # where each sequence's tokens start, N + 1
    context_lens: list[int]
    slots: list[int]
    tables: list[list[int]]      # padded with PADDING_BLOCK


@dataclass(frozen=True)
class ForwardBatch:
    """For [decode(B), decode(C), prefill(A, 300)]: 302 tokens, cu_seqlens_q [0, 1, 2, 302],
    context_lens [512, 47, 300]. Flattened rather than padded, in the paged kernel's layout,
    with the short sequences (decodes) first."""

    input_ids: torch.Tensor     # int32 [T]
    positions: torch.Tensor     # int32 [T], each sequence's own RoPE positions
    cu_seqlens_q: torch.Tensor  # int32 [N + 1]
    context_lens: torch.Tensor  # int32 [N]: how many tokens each sequence attends over
    slot_mapping: torch.Tensor  # int32 [T]: the pool slot each token's K/V goes to
    block_tables: torch.Tensor  # int32 [N, max_blocks]
    host: HostLists | None      # None in a captured graph, which must not read the host
    num_short: tuple[int, int] = (0, 0)  # (sequences, tokens) of the leading short ones
    max_query_len: int | None = None

    @property
    def num_seqs(self) -> int:
        return self.context_lens.shape[0]

    @property
    def last_rows(self) -> torch.Tensor:
        """Each sequence's last token row: the one whose logits are sampled."""
        return self.cu_seqlens_q[1:] - 1

    @classmethod
    def from_lists(cls, host: HostLists, device: torch.device | str) -> ForwardBatch:
        lengths = [b - a for a, b in zip(host.offsets, host.offsets[1:], strict=False)]
        short = 0
        while short < len(lengths) and lengths[short] <= SHORT_QUERY:
            short += 1

        def ints(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        return cls(
            input_ids=ints(host.input_ids),
            positions=ints(host.positions),
            cu_seqlens_q=ints(host.offsets),
            context_lens=ints(host.context_lens),
            slot_mapping=ints(host.slots),
            block_tables=ints(host.tables),
            host=host,
            num_short=(short, host.offsets[short]),
            max_query_len=max(lengths),
        )

    @classmethod
    def from_scheduled(cls, scheduled: list[tuple[Request, int]], manager: BlockManager) -> ForwardBatch:
        """Build from (request, tokens to compute now) pairs whose slots are allocated. Each
        request's tokens start at its num_computed_tokens, so a chunk resumes where the last
        one stopped."""
        input_ids, positions, offsets, context_lens, slots, tables = [], [], [0], [], [], []
        for request, count in scheduled:
            start = request.num_computed_tokens
            input_ids.extend(request.token_ids[start : start + count])
            positions.extend(range(start, start + count))
            offsets.append(offsets[-1] + count)
            context_lens.append(start + count)
            slots.extend(manager.slots(request, count))
            tables.append(request.block_table.block_ids)
        width = max(len(table) for table in tables)
        padded = [table + [PADDING_BLOCK] * (width - len(table)) for table in tables]
        host = HostLists(input_ids, positions, offsets, context_lens, slots, padded)
        return cls.from_lists(host, manager.kv.keys[0].device)


def decode_first(scheduled: list[tuple[Request, int]]) -> None:
    """Order a step's requests short (decodes, verifies) before long (prefill chunks), keeping
    order within each, so the decode kernels take one contiguous slice. In place: sampling,
    verification and commit all follow this order."""
    scheduled.sort(key=lambda pair: pair[1] > SHORT_QUERY)


def run_batch(
    model: nn.Module, kv: PagedKvPool, batch: ForwardBatch, rows: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Logits [rows, V] for one ragged batch, every layer writing and reading kv's pages; rows
    selects the token rows that reach the LM head, None meaning all. A self-draft runs the
    target's first layers over a pool of exactly that many."""
    depth = len(model.model.layers)
    assert depth <= kv.num_layers, f"{depth} layers over a {kv.num_layers}-layer pool"
    with torch.inference_mode(), forward_context(Context(kv=kv, batch=batch)):
        return model.compute_logits(model(batch.input_ids, batch.positions), rows)


def init_distributed(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world_size, rank=rank)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tp_worker(model_name: str, config: EngineConfig, rank: int, port: int) -> None:
    """A tensor-parallel rank above 0: load its shard, then run whatever batch rank 0
    broadcasts until it broadcasts None."""
    from mini_vllm.utils.loader import load

    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    init_distributed(rank, config.tensor_parallel_size, port)
    model, _ = load(model_name, config.backend, device=f"cuda:{rank}")
    runner = ModelRunner(model, config)
    while (message := runner.receive()) is not None:
        host, rows = message
        batch = ForwardBatch.from_lists(host, runner.device)
        runner.run(batch, None if rows is None else torch.tensor(rows, device=runner.device))
    dist.destroy_process_group()


class ModelRunner:
    """Owns the model's KV pool and runs batches through it. The pool takes what is left of
    gpu_memory_utilization after a warmup forward of the largest batch, unless num_blocks is
    given. Pure-decode steps replay a captured CUDA graph on the GPU backends."""

    def __init__(self, model: nn.Module, config: EngineConfig) -> None:
        self.model, self.config = model, config
        backend = get_backend(config.backend)
        for module in model.modules():
            if hasattr(module, "backend"):
                module.backend = backend
        weight = next(model.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.workers: list = []
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.stats = SimpleNamespace(graph_steps=0, eager_steps=0)
        c = model.config
        num_blocks = config.num_blocks or self._blocks_that_fit()
        self.kv = PagedKvPool(
            c.num_hidden_layers, num_blocks, config.block_size, c.num_key_value_heads // tp_size(),
            c.head_dim, dtype=self.dtype, device=self.device, fp8=config.fp8_kv_cache,
        )
        if (self.device.type == "cuda" and backend.supports_graphs and not config.enforce_eager
                and tp_size() == 1):
            self._capture()

    def _pool(self, num_blocks: int) -> PagedKvPool:
        c = self.model.config
        return PagedKvPool(c.num_hidden_layers, num_blocks, self.config.block_size,
                           c.num_key_value_heads // tp_size(), c.head_dim, dtype=self.dtype,
                           device=self.device, fp8=self.config.fp8_kv_cache)

    def _blocks_that_fit(self) -> int:
        """Pages that fit in gpu_memory_utilization of the GPU once the weights are in and the
        largest forward has run: its peak activations stay reserved. Ranks agree on the least."""
        if self.device.type != "cuda":
            raise ValueError("num_blocks is required off the GPU")
        config, block = self.config, self.config.block_size
        tokens = min(config.max_batched_tokens, config.max_model_len)
        pages = -(-tokens // block)
        probe = self._pool(pages)
        host = HostLists([0] * tokens, list(range(tokens)), [0, tokens], [tokens], list(range(tokens)),
                         [list(range(pages))])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_batch(self.model, probe, ForwardBatch.from_lists(host, self.device),
                  torch.tensor([tokens - 1], device=self.device))
        peak = torch.cuda.max_memory_allocated()
        del probe
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        outside_torch = total - free - torch.cuda.memory_reserved()
        c = self.model.config
        element = 1 if config.fp8_kv_cache else self.dtype.itemsize
        kv_heads = c.num_key_value_heads // tp_size()
        per_block = 2 * c.num_hidden_layers * block * kv_heads * c.head_dim * element
        budget = total * config.gpu_memory_utilization - outside_torch - peak
        blocks = torch.tensor(max(1, int(budget // per_block)), device=self.device)
        if tp_size() > 1:
            dist.all_reduce(blocks, op=dist.ReduceOp.MIN)
        return int(blocks)

    @torch.inference_mode()
    def _capture(self) -> None:
        """One graph per batch size, largest first, sharing a memory pool. Each captures the
        forward to hidden states only, routed as a pure decode (one query token, no host lists);
        padded rows write to slot -1, which the kernels skip, and attend over nothing."""
        config = self.config
        max_bs = min(config.cuda_graph_max_bs, config.max_sequences)
        self.sizes = sorted({s for s in (1, 2, 4, 8, *range(16, max_bs + 1, 16)) if s <= max_bs} | {max_bs})
        self.width = -(-config.max_model_len // config.block_size)
        ints = {"dtype": torch.int32, "device": self.device}
        buffers = self.buffers = SimpleNamespace(
            input_ids=torch.zeros(max_bs, **ints), positions=torch.zeros(max_bs, **ints),
            slots=torch.full((max_bs,), -1, **ints), context=torch.zeros(max_bs, **ints),
            tables=torch.full((max_bs, self.width), PADDING_BLOCK, **ints),
            cu=torch.arange(max_bs + 1, **ints),
        )
        self.hidden = torch.zeros(max_bs, self.model.config.hidden_size, dtype=self.dtype, device=self.device)
        pool = None
        try:
            for size in reversed(self.sizes):
                batch = ForwardBatch(
                    buffers.input_ids[:size], buffers.positions[:size], buffers.cu[: size + 1],
                    buffers.context[:size], buffers.slots[:size], buffers.tables[:size],
                    host=None, num_short=(size, size), max_query_len=1,
                )
                with forward_context(Context(kv=self.kv, batch=batch)):
                    self.model(batch.input_ids, batch.positions)  # compile kernels, pick GEMMs
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        self.hidden[:size] = self.model(batch.input_ids, batch.positions)
                pool = pool or graph.pool()
                self.graphs[size] = graph
            torch.cuda.synchronize()
        except Exception as error:  # a backend that cannot be captured still serves, eagerly
            print(f"CUDA graph capture failed, running eagerly: {error!r}")
            self.graphs = {}

    def run(self, batch: ForwardBatch, rows: torch.Tensor | None = None) -> torch.Tensor | None:
        """Logits for one step: replayed from a graph when every sequence decodes one token and
        a graph is large enough, otherwise eagerly. With tensor parallelism, rank 0 first sends
        the batch to every worker."""
        if self.workers:
            message = [(batch.host, None if rows is None else rows.tolist())]
            dist.broadcast_object_list(message, src=0)
        bs = batch.num_seqs
        if (self.graphs and batch.max_query_len == 1 and bs <= self.sizes[-1]
                and batch.block_tables.shape[1] <= self.width):
            size = next(s for s in self.sizes if s >= bs)
            b = self.buffers
            b.input_ids[:bs].copy_(batch.input_ids)
            b.positions[:bs].copy_(batch.positions)
            b.slots[:bs].copy_(batch.slot_mapping)
            b.slots[bs:size].fill_(-1)
            b.context[:bs].copy_(batch.context_lens)
            b.context[bs:size].fill_(0)
            b.tables[:bs, : batch.block_tables.shape[1]].copy_(batch.block_tables)
            self.graphs[size].replay()
            self.stats.graph_steps += 1
            with torch.inference_mode():
                return self.model.compute_logits(self.hidden[:bs], rows)
        self.stats.eager_steps += 1
        return run_batch(self.model, self.kv, batch, rows)

    def receive(self) -> tuple[HostLists, list[int] | None] | None:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        return message[0]

    def close(self) -> None:
        """Stop the tensor-parallel workers."""
        if self.workers:
            dist.broadcast_object_list([None], src=0)
            for worker in self.workers:
                worker.join(timeout=60)
            self.workers = []
            dist.destroy_process_group()
