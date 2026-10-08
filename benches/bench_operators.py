"""Each CUDA kernel against the pure-torch branch of its operator and against torch's own op, at
Qwen3-0.6B shapes in bf16.

    uv run python benches/bench_operators.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch.nn.attention import SDPBackend, sdpa_kernel
from utils import CUDA_BUILT, evidence, grouped_bars, per_call_us, save_plot

from mini_vllm import PagedKvPool, RMSNorm, RoPE, paged_attention, swiglu
from mini_vllm import scaled_dot_product_attention_grouped as sdpa

BF16 = torch.bfloat16
HIDDEN, INTERMEDIATE, Q_HEADS, KV_HEADS, HEAD_DIM = 1024, 3072, 16, 8, 128
DEVICE = "cuda"


def normal(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device=DEVICE, dtype=BF16)


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SDPA's flash backend. A single new token sees its whole context, so that case is unmasked;
    a square prefill is causal, which then matches the bottom-right mask."""
    causal = q.shape[-2] == k.shape[-2]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=HEAD_DIM**-0.5, enable_gqa=True
        )


def cases() -> list[tuple[str, int, object, object, object | None]]:
    """(name, bytes it must move at least, pure, cuda, torch's own or None)."""
    out = []
    for rows in (32, 512):
        x, w = normal(rows, HIDDEN), torch.ones(HIDDEN, device=DEVICE, dtype=BF16)
        out.append((
            f"rms_norm [{rows}, {HIDDEN}]", 2 * x.nbytes,
            lambda x=x, w=w: RMSNorm(HIDDEN, w)(x),
            lambda x=x, w=w: RMSNorm(HIDDEN, w, use_cuda=True)(x),
            lambda x=x, w=w: F.rms_norm(x, (HIDDEN,), w, 1e-6),
        ))

        gate, up = normal(rows, INTERMEDIATE), normal(rows, INTERMEDIATE)
        out.append((
            f"swiglu [{rows}, {INTERMEDIATE}]", 3 * gate.nbytes,
            lambda g=gate, u=up: swiglu(g, u),
            lambda g=gate, u=up: swiglu(g, u, use_cuda=True),
            lambda g=gate, u=up: F.silu(g) * u,
        ))

        device = torch.device(DEVICE)
        pure, cuda = RoPE(HEAD_DIM, 4096, device=device), RoPE(HEAD_DIM, 4096, use_cuda=True, device=device)
        q, positions = normal(1, rows, Q_HEADS, HEAD_DIM), torch.arange(1000, 1000 + rows, device=DEVICE)
        out.append((
            f"rope [1, {rows}, {Q_HEADS}, {HEAD_DIM}]", 2 * q.nbytes,
            lambda q=q, p=positions, rope=pure: rope(q, p),
            lambda q=q, p=positions, rope=cuda: rope(q, p),
            None,
        ))

    for name, length, context in (("decode_attention", 1, 2048), ("flash_prefill", 512, 512)):
        q = normal(1, Q_HEADS, length, HEAD_DIM)
        k, v = normal(1, KV_HEADS, context, HEAD_DIM), normal(1, KV_HEADS, context, HEAD_DIM)
        out.append((
            f"{name} L={length} S={context}", 2 * q.nbytes + k.nbytes + v.nbytes,
            lambda q=q, k=k, v=v: sdpa(q, k, v, mask="causal"),
            lambda q=q, k=k, v=v: sdpa(q, k, v, mask="causal", use_cuda=True),
            lambda q=q, k=k, v=v: torch_attention(q, k, v),
        ))

    pool = PagedKvPool(1, 2048, 16, KV_HEADS, HEAD_DIM, dtype=BF16, device=DEVICE)
    key_pages, value_pages = pool.pages(0)
    batches = (("32 decodes, context 1024", [(1, 1024)] * 32), ("one 512-token chunk", [(512, 512)]))
    for label, shape in batches:
        starts, tables, contexts, used = [0], [], [], 0
        for length, context in shape:
            starts.append(starts[-1] + length)
            blocks = -(-context // 16)
            tables.append(list(range(used, used + blocks)))
            used += blocks
            contexts.append(context)
        table = torch.tensor(tables, dtype=torch.int32, device=DEVICE)
        query = normal(starts[-1], Q_HEADS, HEAD_DIM)
        args = (
            query, key_pages, value_pages, table,
            torch.tensor(starts, dtype=torch.int32, device=DEVICE),
            torch.tensor(contexts, dtype=torch.int32, device=DEVICE),
        )
        read = 2 * sum(contexts) * KV_HEADS * HEAD_DIM * 2
        out.append((
            f"paged_attention, {label}", 2 * query.nbytes + read,
            lambda args=args: paged_attention(*args),
            lambda args=args: paged_attention(*args, use_cuda=True),
            None,
        ))

    slots = torch.arange(0, 32 * 16, 16, dtype=torch.int32, device=DEVICE)
    kv = normal(32, KV_HEADS, HEAD_DIM)
    for fp8 in (False, True):
        pool = PagedKvPool(1, 2048, 16, KV_HEADS, HEAD_DIM, dtype=BF16, device=DEVICE, fp8=fp8)

        def write_once(use_cuda: bool, pool: PagedKvPool = pool) -> None:
            pool.write(0, slots, kv, kv, use_cuda)

        out.append((
            f"{'fp8 ' if fp8 else ''}cache write, 32 tokens into 2048 pages",
            kv.nbytes * (3 if fp8 else 4),
            lambda write_once=write_once: write_once(False),
            lambda write_once=write_once: write_once(True),
            None,
        ))
    return out


def main() -> None:
    torch.manual_seed(0)
    title = "Operators: CUDA kernel vs pure torch vs torch's own"
    with evidence(title):
        ceiling = normal(64 * 2**20)  # 128 MB: a copy's read + write is the bandwidth ceiling
        copy_us = per_call_us(lambda: ceiling * 1, calls=4)
        copy_gbps = 2 * ceiling.nbytes / copy_us / 1e3
        print(f"Copy ceiling: {copy_gbps:.0f} GB/s (read + write of 128 MB; the 5070 Laptop is 384 GB/s).\n")

        print("| op | pure | cuda | speedup | cuda GB/s | torch's own |")
        print("|---|---|---|---|---|---|")
        names, pure_times, cuda_times, own_times = [], [], [], []
        for name, nbytes, pure, cuda, own in cases():
            pure_us = per_call_us(pure)
            cuda_us = per_call_us(cuda) if CUDA_BUILT else None
            own_us = per_call_us(own) if own else None
            print(f"| {name} | {pure_us:.0f} µs | "
                  + (f"{cuda_us:.0f} µs | {pure_us / cuda_us:.1f}x | {nbytes / cuda_us / 1e3:.0f} | "
                     if cuda_us else "— | — | — | ")
                  + (f"{own_us:.0f} µs |" if own_us else "— |"))
            names.append(name)
            pure_times.append(pure_us)
            cuda_times.append(cuda_us if cuda_us else math.nan)
            own_times.append(own_us if own_us else math.nan)
        print("\nGB/s counts the bytes the op must move at least, so it is a floor on what it moved."
              " Per-call times batch 50 calls between two CUDA syncs.")

        figure = Figure(figsize=(10, 8))
        figure.suptitle(title)
        axes = figure.subplots()
        series = {"pure": pure_times, "cuda": cuda_times, "torch's own": own_times}
        grouped_bars(axes, names, series, horizontal=True)
        axes.invert_yaxis()
        axes.set_xscale("log")
        axes.set_xlabel("µs per call (log)")
        save_plot(figure, "bench_operators")


if __name__ == "__main__":
    main()
