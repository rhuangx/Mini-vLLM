"""Latency and throughput from request records, and memory-bandwidth utilization (MBU): the
fraction of the GPU's peak bandwidth a decode step spends reading weights and KV cache.

A decode step reads every weight once plus each sequence's cached keys and values, and little
else, so bytes over step time against the peak says how close the engine runs to the hardware.
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cloud.workload import RequestRecord, Workload

# Peak DRAM bandwidth in bytes/s, by a substring of the device name; the first match wins.
PEAK_BANDWIDTH = [
    ("H100 80GB HBM3", 3.35e12),  # SXM, the only H100 Modal serves
    ("H100 PCIe", 2.0e12),
    ("H100 NVL", 3.9e12),
    ("H200", 4.8e12),
    ("A100-SXM4-80GB", 2.04e12),
    ("RTX 5070 Laptop", 3.84e11),
]


def peak_bandwidth(device: str) -> float | None:
    return next((peak for name, peak in PEAK_BANDWIDTH if name in device), None)


@dataclass(frozen=True)
class Geometry:
    """Bytes a decode step must read: all the weights, and per cached token its K and V."""

    weight_bytes: int          # every parameter a forward touches; the embedding gather is not counted
    kv_bytes_per_token: int    # across all layers, K and V

    @classmethod
    def from_hf_config(cls, config, dtype_bytes: int = 2, kv_bytes: int = 2) -> Geometry:
        e, i, n = config.hidden_size, config.intermediate_size, config.num_hidden_layers
        d = getattr(config, "head_dim", None) or e // config.num_attention_heads
        hq, hk = config.num_attention_heads, config.num_key_value_heads
        attention = e * hq * d + 2 * e * hk * d + hq * d * e + 2 * d  # q, k, v, o, q/k norms
        mlp = 3 * e * i
        per_layer = attention + mlp + 2 * e  # two RMSNorms
        lm_head = config.vocab_size * e      # read in full whether or not it is tied
        params = n * per_layer + e + lm_head
        return cls(params * dtype_bytes, 2 * n * hk * d * kv_bytes)

    def decode_bytes(self, context_tokens: int) -> int:
        return self.weight_bytes + self.kv_bytes_per_token * context_tokens


def _stats(values: Sequence[float], scale: float = 1.0) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(p: float) -> float:
        return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))] * scale

    return {
        "mean": statistics.fmean(ordered) * scale,
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1] * scale,
    }


def summarize(records: list[RequestRecord], wall: float) -> dict:
    """TTFT, time per output token after the first (TPOT), inter-token gaps (ITL), in ms, and
    throughput. TTFT counts from arrival, so queueing behind other requests is included."""
    ttft = [r.token_times[0] - r.arrival for r in records if r.token_times]
    tpot = [
        (r.token_times[-1] - r.token_times[0]) / (len(r.token_times) - 1)
        for r in records
        if len(r.token_times) > 1
    ]
    itl = [b - a for r in records for a, b in zip(r.token_times, r.token_times[1:], strict=False)]
    output_tokens = sum(len(r.token_times) for r in records)
    input_tokens = sum(r.input_len for r in records)
    return {
        "requests": len(records),
        "wall_s": wall,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall,
        "total_tok_s": (output_tokens + input_tokens) / wall,
        "requests_s": len(records) / wall,
        "ttft_ms": _stats(ttft, 1e3),
        "tpot_ms": _stats(tpot, 1e3),
        "itl_ms": _stats(itl, 1e3),
    }


def estimated_decode_mbu(summary: dict, workload: Workload, geometry: Geometry, peak: float) -> float | None:
    """Steady-state MBU from outside the engine, so it applies to both: a closed-loop decode
    workload holds `concurrency` sequences whose context averages input + output / 2, and a
    step takes the median inter-token gap."""
    if workload.concurrency <= 0 or summary["itl_ms"] is None or summary["itl_ms"]["p50"] <= 0:
        return None
    sequences = min(workload.concurrency, workload.num_requests)
    context = sequences * (workload.shared_prefix + workload.input_len + workload.output_len / 2)
    return geometry.decode_bytes(round(context)) / (summary["itl_ms"]["p50"] / 1e3) / peak


def step_mbu(steps: list[tuple[float, object]], geometry: Geometry, peak: float) -> dict | None:
    """Exact MBU over Mini-vLLM's pure-decode steps, from its own StepStats: against the GPU
    time of the forward alone, and against the whole step's wall time (host work included)."""
    forward, wall = [], []
    for seconds, stats in steps:
        if stats is None or not stats.pure_decode:
            continue
        moved = geometry.decode_bytes(stats.context_tokens)
        wall.append(moved / seconds / peak)
        if stats.forward_ms:
            forward.append(moved / (stats.forward_ms / 1e3) / peak)
    if not wall:
        return None
    return {
        "decode_steps": len(wall),
        "forward_mbu": _stats(forward),
        "step_mbu": _stats(wall),
        "step_ms": _stats([s for s, st in steps if st is not None and st.pure_decode], 1e3),
        "forward_ms": _stats([st.forward_ms for _, st in steps if st and st.pure_decode and st.forward_ms]),
    }


def _query(*fields: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [value.strip() for value in out.strip().splitlines()[0].split(",")]


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def python_loop_ms() -> float:
    """Best of three runs of a fixed pure-Python loop: how fast this host runs the interpreter,
    which is what bounds a step that waits on Python. Modal's sandbox hides the CPU model, so
    this is the number that tells two containers apart."""
    import time

    best = float("inf")
    for _ in range(3):
        began = time.perf_counter()
        table: dict[int, int] = {}
        total = 0
        for i in range(500_000):
            table[i & 1023] = i
            total += table[i & 1023]
        best = min(best, time.perf_counter() - began)
    return best * 1e3


def environment() -> dict:
    """The device, the host CPU and the software stack, recorded with every result. A run on the
    wrong GPU or driver is not comparable to the baseline, and an engine whose steps are bound by
    Python overhead runs as fast as the host CPU it landed on: between two Modal containers, such
    numbers have moved by 15-60%, while GPU-bound ones held within 1%."""
    import torch

    props = torch.cuda.get_device_properties(0)
    smi = _query("driver_version", "clocks.max.sm", "clocks.max.mem", "power.limit")
    env = {
        "device": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / 2**30, 1),
        "sms": props.multi_processor_count,
        "cpu": _cpu_model(),
        "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "python_loop_ms": round(python_loop_ms(), 1),
        "driver": smi[0] if smi else None,
        "max_sm_clock": smi[1] if len(smi) > 1 else None,
        "max_mem_clock": smi[2] if len(smi) > 2 else None,
        "power_limit": smi[3] if len(smi) > 3 else None,
        "peak_bandwidth": peak_bandwidth(props.name),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    return env
