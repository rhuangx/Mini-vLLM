"""Turn a sweep's summary.json into markdown tables (and plots), with deltas against a baseline.

    uv run python -m cloud.report results/modal/<tag>/<run>/summary.json --baseline docs/perf/baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["mini-cuda", "mini-triton", "mini-torch", "vllm", "vllm-eager"]


def _labels(results: dict) -> list[str]:
    return sorted(results, key=lambda label: (ORDER.index(label) if label in ORDER else len(ORDER), label))


def _workload(result: dict, name: str) -> dict | None:
    return next((w for w in result.get("workloads", []) if w["workload"]["name"] == name), None)


def _get(entry: dict | None, *path: str) -> float | None:
    value = entry
    for key in path:
        if not isinstance(value, dict) or value.get(key) is None:
            return None
        value = value[key]
    return value


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _delta(now: float | None, then: float | None) -> str:
    if now is None or then is None or then == 0:
        return ""
    return f" ({(now - then) / then:+.0%})"


def _table(results, baseline, names, title, path, digits=1, scale=1.0) -> list[str]:
    labels = _labels(results)
    lines = [f"**{title}**", "", "| workload | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name in names:
        cells = []
        for label in labels:
            now = _get(_workload(results[label], name), *path)
            then = _get(_workload(baseline.get(label, {}), name), *path) if baseline else None
            now = None if now is None else now * scale
            then = None if then is None else then * scale
            cells.append(_fmt(now, digits) + _delta(now, then))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def render(results: dict, baseline: dict | None = None, run: str = "") -> str:
    labels = _labels(results)
    out = [f"# Sweep {run}".rstrip(), ""]
    if baseline:
        out += ["Parenthesized: change against the baseline run.", ""]

    out += ["## Environment", "", "| engine | device | driver | torch | CUDA | load s | KV blocks |",
            "|---|---|---|---|---|---|---|"]
    for label in labels:
        r = results[label]
        if "error" in r:
            out.append(f"| {label} | **failed**: {r['error'][:200]} | | | | | |")
            continue
        env = r.get("env", {})
        version = r.get("vllm_version")
        torch_version = env.get("torch", "") + (f", vllm {version}" if version else "")
        out.append(f"| {label} | {env.get('device')} | {env.get('driver')} | {torch_version} | "
                   f"{env.get('cuda')} | {_fmt(r.get('load_s'))} | {r.get('kv_blocks')} |")
    out.append("")

    names = []
    for label in labels:
        for w in results[label].get("workloads", []):
            if w["workload"]["name"] not in names:
                names.append(w["workload"]["name"])
    groups = {
        "W1 (decode-heavy: 128 in / 512 out, closed loop)": [n for n in names if n.startswith("W1")],
        "W2 (prefill-heavy: 2048 in / 16 out, 8 in flight)": [n for n in names if n == "W2"],
        "W3 (mixed lengths, Poisson arrivals)": [n for n in names if n.startswith("W3")],
        "W4 (1024-token shared prefix + 64 unique)": [n for n in names if n == "W4"],
        "Other": [n for n in names if not n.startswith(("W1", "W2", "W3", "W4"))],
    }
    for title, group in groups.items():
        if not group:
            continue
        out += [f"## {title}", ""]
        out += _table(results, baseline, group, "Output tokens/s", ("summary", "output_tok_s"), 0)
        out += _table(results, baseline, group, "TTFT p50 ms", ("summary", "ttft_ms", "p50"))
        out += _table(results, baseline, group, "TTFT p99 ms", ("summary", "ttft_ms", "p99"))
        out += _table(results, baseline, group, "TPOT p50 ms", ("summary", "tpot_ms", "p50"), 2)
        out += _table(results, baseline, group, "ITL p99 ms", ("summary", "itl_ms", "p99"), 2)
        if title.startswith("W1"):
            out += _table(results, baseline, group, "Decode MBU % (estimated from ITL)",
                          ("mbu_estimate",), 1, 100)
            out += _table(results, baseline, group, "Decode MBU % (Mini-vLLM, forward GPU time, p50)",
                          ("steps", "forward_mbu", "p50"), 1, 100)
            out += _table(results, baseline, group, "Decode MBU % (Mini-vLLM, whole step, p50)",
                          ("steps", "step_mbu", "p50"), 1, 100)

    for label in labels:
        for name, p in results[label].get("profiles", {}).items():
            out += [f"## Profile: {label} / {name}", "",
                    f"GPU {_fmt(p['gpu_ms_per_step'], 2)} ms/step, "
                    f"{_fmt(p['kernels_per_step'], 0)} kernels/step "
                    f"over {p['active_steps']} steps.", ""]
            out += ["| GPU time by category | ms/step |", "|---|---|"]
            out += [f"| {k} | {v:.3f} |" for k, v in p["categories_ms_per_step"].items()]
            out += ["", "| host span | ms/step |", "|---|---|"]
            out += [f"| {k} | {v:.3f} |" for k, v in p["spans_ms_per_step"].items()]
            out += ["", "| top kernel | category | ms/step | calls/step |", "|---|---|---|---|"]
            out += [
                f"| `{k['kernel'][:80]}` | {k['category']} | {k['ms_per_step']:.3f} "
                f"| {k['calls_per_step']:g} |"
                for k in p["top_kernels"][:10]
            ]
            out.append("")
    return "\n".join(out)


def plot(results: dict, out_dir: Path) -> None:
    """Throughput and TPOT against concurrency for W1, one line per engine."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for label in _labels(results):
        points = sorted(
            (w["workload"]["concurrency"], w["summary"]["output_tok_s"], w["summary"]["tpot_ms"]["p50"])
            for w in results[label].get("workloads", [])
            if w["workload"]["name"].startswith("W1") and w["summary"]["tpot_ms"]
        )
        if points:
            left.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
            right.plot([p[0] for p in points], [p[2] for p in points], marker="o", label=label)
    for axes, ylabel in ((left, "output tokens/s"), (right, "TPOT p50 (ms)")):
        axes.set_xscale("log", base=2)
        axes.set_xlabel("requests in flight")
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.3)
        axes.legend()
    figure.suptitle("W1: Qwen3-8B decode-heavy, H100")
    figure.savefig(out_dir / "w1.png", dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, help="default: report.md next to the summary")
    args = parser.parse_args()
    results = json.loads(args.summary.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    run = args.summary.parent.name
    text = render(results, baseline, run)
    out = args.out or args.summary.with_name("report.md")
    out.write_text(text)
    plot(results, out.parent)
    print(text)


if __name__ == "__main__":
    main()
