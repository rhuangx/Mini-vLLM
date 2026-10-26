"""Turn a sweep's summary.json into markdown tables (and a plot), with deltas against a baseline.

    uv run python -m cloud.report results/modal/<tag>/<run>/summary.json --baseline docs/perf/baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["mini-cuda", "mini-triton", "mini-torch", "vllm", "vllm-eager"]
BACKENDS = {"cuda": "CUDA kernels", "triton": "Triton kernels", "torch": "pure PyTorch"}


def _labels(results: dict) -> list[str]:
    return sorted(results, key=lambda label: (ORDER.index(label) if label in ORDER else len(ORDER), label))


def engine_name(label: str, result: dict) -> str:
    """What a result's engine is, in words: "Mini-vLLM, CUDA kernels", "vLLM 0.31.0, eager"."""
    if result.get("engine") == "vllm" or label.startswith("vllm"):
        version = result.get("vllm_version", "")
        eager = result.get("enforce_eager", label.endswith("eager"))
        return f"vLLM {version}".rstrip() + (", eager" if eager else "")
    backend = result.get("backend") or label.removeprefix("mini-")
    return f"Mini-vLLM, {BACKENDS.get(backend, backend)}"


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


def _table(results, baseline, rows, title, path, digits=1, scale=1.0) -> list[str]:
    """One metric: a row per workload in `rows` (name, row label), a column per engine."""
    labels = _labels(results)
    header = " | ".join(engine_name(label, results[label]) for label in labels)
    lines = [f"**{title}**", "", f"| load | {header} |", "|---|" + "---|" * len(labels)]
    for name, row in rows:
        cells = []
        for label in labels:
            now = _get(_workload(results[label], name), *path)
            then = _get(_workload(baseline.get(label, {}), name), *path) if baseline else None
            now = None if now is None else now * scale
            then = None if then is None else then * scale
            cells.append(_fmt(now, digits) + _delta(now, then))
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def render(results: dict, baseline: dict | None = None, run: str = "") -> str:
    labels = _labels(results)
    out = [f"# Sweep {run}".rstrip(), ""]
    if baseline:
        out += ["Parenthesized: change against the baseline run.", ""]

    out += ["## Environment", "",
            "| engine | device | host CPU | driver | torch | CUDA | load s | KV blocks |",
            "|---|---|---|---|---|---|---|---|"]
    for label in labels:
        r = results[label]
        name = engine_name(label, r)
        if "error" in r:
            out.append(f"| {name} | **failed**: {r['error'][:200]} | | | | | | |")
            continue
        env = r.get("env", {})
        cpu = f"{env['cpu']} ({env.get('cpus')} cores)" if env.get("cpu") else "not recorded"
        out.append(f"| {name} | {env.get('device')} | {cpu} | {env.get('driver')} | {env.get('torch')} | "
                   f"{env.get('cuda')} | {_fmt(r.get('load_s'))} | {r.get('kv_blocks')} |")
    out.append("")

    # One section per kind of workload, in the order the sweep ran them; a row per load.
    kinds: dict[str, dict] = {}
    for label in labels:
        for entry in results[label].get("workloads", []):
            spec = entry["workload"]
            kind = kinds.setdefault(spec.get("kind", spec["name"]), {"description": spec.get("description"),
                                                                      "rows": []})
            row = (spec["name"], spec.get("load", spec["name"]))
            if row not in kind["rows"]:
                kind["rows"].append(row)
    for kind, section in kinds.items():
        rows = section["rows"]
        heading = kind.capitalize()
        out += [f"## {heading}" + (f": {section['description']}" if section["description"] else ""), ""]
        out += _table(results, baseline, rows, "Output tokens/s", ("summary", "output_tok_s"), 0)
        out += _table(results, baseline, rows, "Time to first token, p50 ms", ("summary", "ttft_ms", "p50"))
        out += _table(results, baseline, rows, "Time to first token, p99 ms", ("summary", "ttft_ms", "p99"))
        out += _table(results, baseline, rows, "Time per output token, p50 ms",
                      ("summary", "tpot_ms", "p50"), 2)
        out += _table(results, baseline, rows, "Inter-token latency, p99 ms", ("summary", "itl_ms", "p99"), 2)
        if kind == "decode-heavy":
            out += _table(results, baseline, rows,
                          "Decode memory-bandwidth utilization %, estimated from ITL",
                          ("mbu_estimate",), 1, 100)
            out += _table(results, baseline, rows,
                          "Decode memory-bandwidth utilization %, Mini-vLLM's own forward GPU time (p50)",
                          ("steps", "forward_mbu", "p50"), 1, 100)
            out += _table(results, baseline, rows,
                          "Decode memory-bandwidth utilization %, Mini-vLLM's whole step (p50)",
                          ("steps", "step_mbu", "p50"), 1, 100)

    for label in labels:
        for name, p in results[label].get("profiles", {}).items():
            out += [f"## Profile: {engine_name(label, results[label])}, {name.replace('-', ' ')}", "",
                    f"A step takes {_fmt(p.get('step_ms'), 2)} ms, of which the GPU runs kernels for "
                    f"{_fmt(p['gpu_ms_per_step'], 2)} ms ({_fmt(100 * (p.get('gpu_busy') or 0), 0)}% busy); "
                    f"{_fmt(p['kernels_per_step'], 0)} kernels per step, over {p['active_steps']} steps.", ""]
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


def plot(results: dict, out_dir: Path) -> Path | None:
    """Throughput and time per output token against requests in flight, for the decode-heavy
    workload, one line per engine. Returns the file written, if any."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        return None
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    description = None
    for label in _labels(results):
        decode_heavy = [w for w in results[label].get("workloads", [])
                        if w["workload"].get("kind") == "decode-heavy"
                        and w.get("summary", {}).get("tpot_ms")]
        points = sorted((w["workload"]["concurrency"], w["summary"]["output_tok_s"],
                         w["summary"]["tpot_ms"]["p50"]) for w in decode_heavy)
        if points:
            description = description or decode_heavy[0]["workload"].get("description")
            name = engine_name(label, results[label])
            left.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
            right.plot([p[0] for p in points], [p[2] for p in points], marker="o", label=name)
    if description is None:
        plt.close(figure)
        return None
    right.set_yscale("log")  # the pure-PyTorch path sits an order of magnitude above the rest
    for axes, ylabel in ((left, "output tokens/s"), (right, "time per output token, p50 (ms)")):
        axes.set_xscale("log", base=2)
        axes.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:g}"))  # 1, 2, 4, not 2^0
        axes.set_xlabel("requests in flight")
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.3)
        axes.legend()
    figure.suptitle(f"Qwen3-8B on one H100, decode-heavy: {description}")
    path = out_dir / "decode-heavy.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


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
