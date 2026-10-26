"""The Modal app: Mini-vLLM's tests, benches, workload sweeps and profiles on H100s, and vLLM
on the same workloads as the baseline. Run from the repository root:

    uv run --group cloud modal run -m cloud.app::download            # once: models into the volume
    uv run --group cloud modal run -m cloud.app::probe               # driver, versions, kernels work
    uv run --group cloud modal run -m cloud.app::tests               # the full pytest suite
    uv run --group cloud modal run -m cloud.app::bench --script bench_operators.py
    uv run --group cloud modal run -m cloud.app::sweep --suite quick --engines mini-cuda,vllm --tag phase1

Results go to the mini-vllm-results volume under <tag>/<time>-<commit>/, and a sweep's summary
and report also to results/modal/ locally. `gpu="H100!"` pins an H100: plain "H100" may be
served by an H200, whose bandwidth would make every MBU figure wrong.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

from cloud import report
from cloud.adapters import ENGINE
from cloud.images import HF_HOME, REMOTE, REPO, RESULTS, VLLM_VERSION, mini_image, vllm_image
from cloud.workload import PROFILES, SUITES

app = modal.App("mini-vllm")
hf_cache = modal.Volume.from_name("mini-vllm-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("mini-vllm-results", create_if_missing=True)
VOLUMES = {HF_HOME: hf_cache, RESULTS: results_volume}
GPU = "H100!"
MINI = mini_image()
VLLM = vllm_image()
MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B")  # the profiled model, and the one the tests use


def _save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _stream(command: list[str], log: Path, env: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """Run command in the checkout, echoing its output to the Modal log and to log; returns the
    exit code and the last lines."""
    log.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log.open("w") as sink, subprocess.Popen(
        command, cwd=REMOTE, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sink.write(line)
            tail = [*tail[-39:], line.rstrip()]
    return process.returncode, tail


# Remote functions.

@app.function(image=MINI, volumes={HF_HOME: hf_cache}, timeout=3600, cpu=4.0)
def download_models(models: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for model in models:
        path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        print(f"{model} -> {path}")
    hf_cache.commit()


@app.function(image=MINI, gpu=GPU, volumes=VOLUMES, timeout=900, cpu=4.0)
def env_probe(run_dir: str) -> dict:
    from cloud import metrics, probe

    info = {
        **metrics.environment(),
        "python_executable": sys.executable,
        "extension": probe.extension(),
        "triton_jit": probe.triton_jit(),
        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
    }
    print(json.dumps({k: v for k, v in info.items() if k != "nvidia_smi"}, indent=2))
    print(info["nvidia_smi"])
    _save(Path(RESULTS) / run_dir / "env.json", info)
    results_volume.commit()
    return info


@app.function(image=MINI, gpu=GPU, volumes=VOLUMES, timeout=3600, cpu=8.0, memory=32768)
def run_pytest(run_dir: str, args: str) -> dict:
    out = Path(RESULTS) / run_dir
    command = [sys.executable, "-m", "pytest", "-ra", "-p", "no:cacheprovider",
               f"--junitxml={out / 'junit.xml'}", *shlex.split(args)]
    code, tail = _stream(command, out / "pytest.log")
    results_volume.commit()
    return {"returncode": code, "tail": tail}


@app.function(image=MINI, gpu=GPU, volumes=VOLUMES, timeout=3600, cpu=8.0, memory=65536)
def run_bench(run_dir: str, commit: str, script: str, args: str) -> dict:
    out = Path(RESULTS) / run_dir / "benches"
    env = {"MINI_VLLM_RESULTS": str(out), "MINI_VLLM_COMMIT": commit}
    code, tail = _stream([sys.executable, f"benches/{script}", *shlex.split(args)],
                         out / f"{Path(script).stem}.log", env)
    results_volume.commit()
    return {"returncode": code, "tail": tail}


def _measure(label, factory_fresh, release, workloads, geometry, peak, out: Path) -> list[dict]:
    """Run each workload on a fresh adapter; a workload that fails is recorded, not fatal."""
    from cloud import metrics
    from cloud import workload as wl

    entries = []
    for spec in workloads:
        workload = wl.Workload(**spec)
        adapter = factory_fresh()
        try:
            records, wall = wl.run(adapter, workload)
        except Exception as error:
            entries.append({"workload": spec, "error": repr(error)})
            print(f"[{label}] {workload.name}: FAILED {error!r}")
            continue
        finally:
            release(adapter)
        summary = metrics.summarize(records, wall)
        steps = getattr(adapter, "steps", None)
        entries.append({
            "workload": spec,
            "summary": summary,
            "mbu_estimate": metrics.estimated_decode_mbu(summary, workload, geometry, peak) if peak else None,
            "steps": metrics.step_mbu(steps, geometry, peak) if steps and peak else None,
        })
        with (out / f"{workload.name}.jsonl").open("w") as f:
            f.writelines(json.dumps(record.to_json()) + "\n" for record in records)
        ttft, tpot = summary["ttft_ms"], summary["tpot_ms"]
        print(f"[{label}] {workload.name}: {summary['output_tok_s']:.0f} tok/s, "
              f"TTFT p50 {ttft['p50']:.1f} ms, TPOT p50 {tpot['p50'] if tpot else float('nan'):.2f} ms")
    return entries


@app.function(image=MINI, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, cpu=8.0, memory=65536)
def sweep_mini(run_dir: str, label: str, backend: str, workloads: list[dict], profiles: list[str]) -> dict:
    from transformers import AutoConfig

    from cloud import adapters, metrics, profile_engine
    from mini_vllm.utils import profiling

    profiling.enable()  # spans for the traces, and CUDA-event forward times for the exact MBU
    out = Path(RESULTS) / run_dir / label
    out.mkdir(parents=True, exist_ok=True)
    env = metrics.environment()
    geometry = metrics.Geometry.from_hf_config(AutoConfig.from_pretrained(ENGINE["model"]))
    factory = adapters.MiniVllmFactory(backend)
    result = {
        "label": label, "engine": "mini-vllm", "backend": backend, "env": env, "load_s": factory.load_s,
        "kv_blocks": ENGINE["num_blocks"], "geometry": asdict(geometry), "engine_config": ENGINE,
        "workloads": _measure(label, factory.fresh, factory.release, workloads, geometry,
                              env["peak_bandwidth"], out),
        "profiles": {},
    }
    for name in profiles:
        adapter = factory.fresh()
        try:
            trace_dir = out / "profiles" / name
            result["profiles"][name] = profile_engine.profile(adapter, PROFILES[name], trace_dir)
        finally:
            factory.release(adapter)
    _save(out / "result.json", result)
    results_volume.commit()
    return result


@app.function(image=VLLM, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, cpu=8.0, memory=65536)
def sweep_vllm(run_dir: str, label: str, enforce_eager: bool, workloads: list[dict]) -> dict:
    from transformers import AutoConfig

    from cloud import adapters, metrics

    out = Path(RESULTS) / run_dir / label
    out.mkdir(parents=True, exist_ok=True)
    env = metrics.environment()
    geometry = metrics.Geometry.from_hf_config(AutoConfig.from_pretrained(ENGINE["model"]))
    engine, load_s = adapters.vllm_engine(enforce_eager)
    result = {
        "label": label, "engine": "vllm", "vllm_version": VLLM_VERSION, "enforce_eager": enforce_eager,
        "env": env, "load_s": load_s, "kv_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
        "geometry": asdict(geometry), "engine_config": ENGINE,
        "workloads": _measure(label, lambda: adapters.VllmAdapter(engine), lambda adapter: None,
                              workloads, geometry, env["peak_bandwidth"], out),
    }
    _save(out / "result.json", result)
    results_volume.commit()
    return result


# Local entry points.

def _run_dir(tag: str) -> tuple[str, str]:
    """<tag>/<time>-<commit>, and the commit (suffixed -dirty with uncommitted changes)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "nogit"
    if git("status", "--porcelain"):
        commit += "-dirty"
    return f"{tag}/{time.strftime('%Y%m%d-%H%M%S')}-{commit}", commit


@app.local_entrypoint()
def download(models: str = ",".join(MODELS)) -> None:
    download_models.remote(models.split(","))


@app.local_entrypoint()
def probe(tag: str = "probe") -> None:
    run_dir, _ = _run_dir(tag)
    info = env_probe.remote(run_dir)
    local = REPO / "results" / "modal" / run_dir
    _save(local / "env.json", info)
    print(f"\n{info['device']} | driver {info['driver']} | torch {info['torch']} (CUDA {info['cuda']}) | "
          f"extension: {info['extension']} | triton: {info['triton_jit']}\nsaved {local / 'env.json'}")


@app.local_entrypoint()
def tests(args: str = "", tag: str = "tests") -> None:
    run_dir, _ = _run_dir(tag)
    outcome = run_pytest.remote(run_dir, args)
    print("\n".join(outcome["tail"][-15:]))
    print(f"\npytest exited {outcome['returncode']}; "
          f"log and junit.xml in volume mini-vllm-results:/{run_dir}")
    if outcome["returncode"] != 0:
        sys.exit(outcome["returncode"])


@app.local_entrypoint()
def bench(script: str, args: str = "", tag: str = "benches") -> None:
    run_dir, commit = _run_dir(tag)
    outcome = run_bench.remote(run_dir, commit, script, args)
    print(f"\n{script} exited {outcome['returncode']}; plots in volume mini-vllm-results:/{run_dir}/benches")


@app.local_entrypoint()
def sweep(suite: str = "quick", engines: str = "mini-cuda,vllm", tag: str = "adhoc",
          profile: bool = True) -> None:
    """Fan the engines out to one H100 each, in parallel, then write summary.json and report.md
    under results/modal/<run>. mini-torch runs the reduced "torch" suite: its pure-Python
    paged attention is the slowest path and two points are its baseline."""
    run_dir, _ = _run_dir(tag)
    calls = {}
    for label in engines.split(","):
        if label.startswith("mini-"):
            backend = label.removeprefix("mini-")
            torch_only = backend == "torch" and suite not in ("smoke", "none")
            names = SUITES["torch"] if torch_only else SUITES[suite]
            # At ~0.5 s a step, the pure path is not traced at 256 in flight.
            traced = [name for name in PROFILES if backend != "torch" or name != "decode-c256"]
            calls[label] = sweep_mini.spawn(run_dir, label, backend, [asdict(w) for w in names],
                                            traced if profile else [])
        elif label in ("vllm", "vllm-eager"):
            if not SUITES[suite]:
                continue
            calls[label] = sweep_vllm.spawn(run_dir, label, label == "vllm-eager",
                                            [asdict(w) for w in SUITES[suite]])
        else:
            raise ValueError(f"unknown engine {label!r}: mini-<backend>, vllm or vllm-eager")

    results = {}
    for label, call in calls.items():
        try:
            results[label] = call.get()
        except Exception as error:
            results[label] = {"label": label, "error": repr(error)}
            print(f"{label} failed: {error!r}")

    local = REPO / "results" / "modal" / run_dir
    _save(local / "summary.json", results)
    text = report.render(results, run=run_dir)
    (local / "report.md").write_text(text)
    report.plot(results, local)
    print(text)
    print(f"\nsaved {local}/summary.json and report.md; traces: "
          f"modal volume get mini-vllm-results /{run_dir} results/modal/{run_dir}")
