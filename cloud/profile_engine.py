"""Trace a window of engine steps with torch.profiler and say where the time went: the GPU's,
by kernel category, and the host's, by the engine's own spans (utils/profiling.py).

Writes trace.json.gz (open in Perfetto or chrome://tracing), ops.txt (the top ops by GPU time)
and kernels.csv (every kernel with its category), and returns the summary.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from cloud import workload as wl

# Kernel name patterns, tried in order; the first match is the kernel's category.
CATEGORIES = [
    ("attention", r"paged_attention|decode_attention|flash|fmha|attn|sdpa"),
    ("kv_write", r"cache_update|kv_quantize|store_kv|reshape_and_cache|index_copy|scatter"),
    ("gemm", r"gemm|cutlass|xmma|nvjet|cublas|matmul|splitk|Kernel2"),
    ("norm", r"rms_norm|norm"),
    ("rope", r"rope|rotary"),
    ("activation", r"swiglu|silu|act_and_mul|gelu"),
    ("sampling", r"softmax|argmax|topk|sort|multinomial|exponential|cumsum|distribution"),
    ("copy", r"memcpy|memset|copy|cat|fill|gather|index|Memcpy|Memset"),
    ("elementwise", r"elementwise|vectorized|reduce"),
]
SPANS = ("schedule", "prepare", "forward", "sample", "spec.propose", "spec.verify", "commit",
         "draft.prepare", "draft.forward", "draft.sample")


def categorize(kernel: str) -> str:
    return next((name for name, pattern in CATEGORIES if re.search(pattern, kernel, re.IGNORECASE)), "other")


def profile(adapter: wl.Adapter, workload: wl.Workload, out_dir: Path, wait: int = 4, warmup: int = 2,
            active: int = 10) -> dict:
    """Run the whole workload, tracing steps wait + warmup .. wait + warmup + active."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, schedule

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(out_dir / "trace.json.gz")),
    ) as prof:
        wl.run(adapter, workload, after_step=prof.step)

    events = prof.events() or []
    kernel_us: dict[str, float] = defaultdict(float)
    kernel_calls: dict[str, int] = defaultdict(int)
    span_us: dict[str, float] = defaultdict(float)
    for event in events:
        if event.device_type == DeviceType.CUDA:
            kernel_us[event.name] += event.time_range.elapsed_us()
            kernel_calls[event.name] += 1
        elif event.name in SPANS:
            span_us[event.name] += event.cpu_time_total

    by_category: dict[str, float] = defaultdict(float)
    with (out_dir / "kernels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "kernel", "calls", "total_us", "us_per_step"])
        for name, us in sorted(kernel_us.items(), key=lambda item: -item[1]):
            category = categorize(name)
            by_category[category] += us
            writer.writerow([category, name, kernel_calls[name], round(us, 1), round(us / active, 1)])
    (out_dir / "ops.txt").write_text(
        prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
    )

    gpu_us = sum(by_category.values())
    categories = sorted(by_category.items(), key=lambda item: -item[1])
    top = sorted(kernel_us.items(), key=lambda item: -item[1])[:15]
    return {
        "workload": workload.name,
        "active_steps": active,
        "gpu_ms_per_step": gpu_us / active / 1e3,
        "kernels_per_step": sum(kernel_calls.values()) / active,
        "categories_ms_per_step": {
            name: round(us / active / 1e3, 3) for name, us in categories
        },
        "spans_ms_per_step": {
            name: round(span_us[name] / active / 1e3, 3) for name in SPANS if name in span_us
        },
        "top_kernels": [
            {"kernel": name[:120], "category": categorize(name), "ms_per_step": round(us / active / 1e3, 3),
             "calls_per_step": kernel_calls[name] / active}
            for name, us in top
        ],
    }
