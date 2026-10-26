"""Trace a window of engine steps with torch.profiler and say where the time went: the GPU's,
by kernel category, and the host's, by the engine's own spans (utils/profiling.py).

Writes trace.json.gz (open in Perfetto or chrome://tracing), ops.txt (the top ops by GPU time)
and kernels.csv (every kernel with its category), and returns the summary. The summary is read
back from the exported trace, by event category: torch.profiler's own event list also carries
each span a second time as a GPU-side annotation, which would count as a kernel.

    uv run python -m cloud.profile_engine results/modal/<tag>/<run>   # re-summarize a downloaded run
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from cloud import workload as wl

# Kernel name patterns, tried in order; the first match is the kernel's category.
CATEGORIES = [
    ("attention", r"paged_attention|decode_attention|flash|fmha|attn|sdpa"),
    ("kv_write", r"cache_update|kv_quantize|store_kv|reshape_and_cache|index_copy|scatter"),
    ("gemm", r"gemm|cutlass|xmma|nvjet|cublas|matmul|splitk|gemv|Kernel2"),
    ("norm", r"rms_norm|norm"),
    ("rope", r"rope|rotary"),
    ("activation", r"swiglu|silu|act_and_mul|gelu"),
    ("sampling", r"softmax|argmax|topk|sort|multinomial|exponential|cumsum|distribution"),
    ("copy", r"memcpy|memset|copy|cat|fill|gather|index"),
    ("elementwise", r"elementwise|vectorized|reduce"),
]
SPANS = ("schedule", "prepare", "forward", "sample", "spec.propose", "spec.verify", "commit",
         "draft.prepare", "draft.forward", "draft.sample")
GPU_WORK = ("kernel", "gpu_memcpy", "gpu_memset")


def categorize(kernel: str) -> str:
    return next((name for name, pattern in CATEGORIES if re.search(pattern, kernel, re.IGNORECASE)), "other")


def profile(adapter: wl.Adapter, workload: wl.Workload, out_dir: Path, wait: int = 18, warmup: int = 2,
            active: int = 10) -> dict:
    """Run the whole workload, tracing steps wait + warmup .. wait + warmup + active."""
    import torch
    from torch.profiler import ProfilerActivity, schedule

    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / "trace.json.gz"
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace)),
    ) as prof:
        wl.run(adapter, workload, after_step=prof.step)
    (out_dir / "ops.txt").write_text(
        prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
    )
    return {"workload": workload.name, **summarize_trace(trace)}


def summarize_trace(trace: Path) -> dict:
    """GPU time per step by kernel category, host time per step by span, and how busy the GPU
    was: kernel time over step time, which falls well below 1 when launches are the bottleneck.
    Writes kernels.csv next to the trace."""
    with gzip.open(trace) as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    steps = [e["dur"] for e in events
             if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep")]
    active = max(1, len(steps))
    kernel_us: dict[str, float] = defaultdict(float)
    kernel_calls: dict[str, int] = defaultdict(int)
    span_us: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("cat") in GPU_WORK:
            kernel_us[event["name"]] += event.get("dur", 0)
            kernel_calls[event["name"]] += 1
        elif event.get("cat") == "user_annotation" and event["name"] in SPANS:
            span_us[event["name"]] += event.get("dur", 0)

    by_category: dict[str, float] = defaultdict(float)
    ranked = sorted(kernel_us.items(), key=lambda item: -item[1])
    with (trace.parent / "kernels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "kernel", "calls", "total_us", "us_per_step"])
        for name, us in ranked:
            category = categorize(name)
            by_category[category] += us
            writer.writerow([category, name, kernel_calls[name], round(us, 1), round(us / active, 1)])

    step_ms = sum(steps) / active / 1e3
    gpu_ms = sum(kernel_us.values()) / active / 1e3
    categories = sorted(by_category.items(), key=lambda item: -item[1])
    return {
        "active_steps": active,
        "step_ms": step_ms,
        "gpu_ms_per_step": gpu_ms,
        "gpu_busy": gpu_ms / step_ms if step_ms else None,
        "kernels_per_step": sum(kernel_calls.values()) / active,
        "categories_ms_per_step": {name: round(us / active / 1e3, 3) for name, us in categories},
        "spans_ms_per_step": {
            name: round(span_us[name] / active / 1e3, 3) for name in SPANS if name in span_us
        },
        "top_kernels": [
            {"kernel": name[:120], "category": categorize(name), "ms_per_step": round(us / active / 1e3, 3),
             "calls_per_step": kernel_calls[name] / active}
            for name, us in ranked[:15]
        ],
    }


def resummarize(run: Path) -> None:
    """Recompute every profile in a downloaded run from its traces, and rewrite its summary.json
    and report.md: for traces taken before this module's summary changed."""
    from cloud import report

    summary_path = run / "summary.json"
    results = json.loads(summary_path.read_text())
    for label, result in results.items():
        for name, old in result.get("profiles", {}).items():
            trace = run / label / "profiles" / name / "trace.json.gz"
            if trace.is_file():
                result["profiles"][name] = {"workload": old.get("workload"), **summarize_trace(trace)}
                print(f"{label}/{name}: {result['profiles'][name]['gpu_ms_per_step']:.2f} GPU ms per "
                      f"{result['profiles'][name]['step_ms']:.2f} ms step")
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    (run / "report.md").write_text(report.render(results, run=f"{run.parent.name}/{run.name}"))


if __name__ == "__main__":
    resummarize(Path(sys.argv[1]))
