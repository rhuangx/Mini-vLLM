"""Shared by every bench: CUDA timing, the evidence header, the clock sampler, the models every
engine in a run shares, and the plots written to results/."""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm import LLM, EngineConfig, StreamUpdate, from_hf
from mini_vllm.utils.tokenizer import Tokenizer, eos_token_ids

try:
    import mini_vllm_ext  # noqa: F401

    CUDA_BUILT = True
except (ImportError, OSError):  # OSError: the package is there but _ext.so is not built
    CUDA_BUILT = False

# MINI_VLLM_RESULTS moves the plots, e.g. onto a Modal volume; by default they land in results/.
RESULTS = Path(os.environ.get("MINI_VLLM_RESULTS") or Path(__file__).resolve().parent.parent / "results")


def per_call_us(fn: Callable[[], Any], calls: int = 50, repeats: int = 20) -> float:
    """Median microseconds per call. Each repeat brackets `calls` launches with a CUDA sync,
    so the launch overhead is amortized and the clock includes the GPU work."""
    for _ in range(3):
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / calls)
    return statistics.median(times) * 1e6


def percentile(values: list[float], p: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def token_times(updates: Iterator[StreamUpdate]) -> dict[int, list[float]]:
    """Seconds from the start of a stream to each of its tokens, per prompt: a prompt's first
    time is its TTFT and the gaps between the rest are its inter-token latency. Pass the
    stream unstarted, since the clock starts here."""
    start = time.perf_counter()
    times = defaultdict(list)
    for update in updates:
        times[update.index].append(time.perf_counter() - start)
    return times


def _run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout


@dataclass(frozen=True)
class GpuState:
    """What the GPU was doing, as opposed to what it is capable of."""

    name: str
    sm_clock: int
    sm_clock_max: int
    memory_clock: int
    memory_clock_max: int

    @property
    def sm_fraction(self) -> float:
        return self.sm_clock / self.sm_clock_max if self.sm_clock_max else 1.0

    @property
    def memory_fraction(self) -> float:
        return self.memory_clock / self.memory_clock_max if self.memory_clock_max else 1.0

    @property
    def is_throttled(self) -> bool:
        """True below half the rated clocks, where the card is idling rather than working."""
        return self.sm_fraction < 0.5 or self.memory_fraction < 0.5

    def describe(self) -> str:
        return (
            f"{self.name}: sm {self.sm_clock}/{self.sm_clock_max} MHz ({self.sm_fraction:.0%}), "
            f"mem {self.memory_clock}/{self.memory_clock_max} MHz ({self.memory_fraction:.0%})"
        )


def gpu_state() -> GpuState | None:
    """Read the current clocks from nvidia-smi, or None if that is not possible."""
    fields = ["name", "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem"]
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = [value.strip() for value in output.strip().splitlines()[0].split(",")]
    if len(values) != len(fields):
        return None

    def megahertz(text: str) -> int:
        digits = "".join(character for character in text if character.isdigit())
        return int(digits) if digits else 0

    return GpuState(
        values[0], megahertz(values[1]), megahertz(values[2]), megahertz(values[3]), megahertz(values[4])
    )


class ClockSampler:
    """Watches the GPU clocks in the background and keeps the highest seen."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak: GpuState | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            state = gpu_state()
            if state is not None and (self.peak is None or state.sm_clock > self.peak.sm_clock):
                self.peak = state
            self._stop.wait(self.interval)

    def __enter__(self) -> ClockSampler:
        if torch.cuda.is_available():
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exception: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


@contextmanager
def evidence(title: str) -> Iterator[None]:
    """Print the device, the toolchain and TF32, run the bench, then say whether the clocks
    stayed up. A throttled run is not an absolute measurement."""
    props = torch.cuda.get_device_properties(0)
    # A container has no .git, so whoever launched it says which commit this is.
    commit = os.environ.get("MINI_VLLM_COMMIT") or _run("git", "rev-parse", "--short", "HEAD").strip()
    dirty = ""
    if "MINI_VLLM_COMMIT" not in os.environ and _run("git", "status", "--porcelain").strip():
        dirty = " + uncommitted changes"
    driver = _run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").strip()
    print(f"# {title}\n")
    print(f"- {props.name}, sm_{props.major}{props.minor}, {props.total_memory / 2**30:.1f} GB, "
          f"driver {driver or 'unknown'}")
    print(f"- mini-vllm {commit}{dirty}; torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"Python {platform.python_version()}")
    print(f"- CUDA extension {'built' if CUDA_BUILT else 'not built'}; "
          f"TF32 matmul {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}")
    print(f"- {time.strftime('%Y-%m-%d %H:%M')}\n")

    with ClockSampler() as clocks:
        yield
    if clocks.peak is None:
        print("\nClocks not observed (no nvidia-smi reading) — treat the numbers as unverified.")
    else:
        print(f"\nPeak clocks: {clocks.peak.describe()}")
        if clocks.peak.is_throttled:
            print("**THROTTLED** — these numbers are invalid as absolute figures.")
        else:
            print("Clocks stayed at least half of rated; no throttling detected.")


def load_hf(model: str) -> tuple[Any, Tokenizer]:
    """The transformers model and wrapped tokenizer the baselines run and every engine borrows
    weights from. EOS is disabled, so every engine runs every request to max_tokens."""
    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer = Tokenizer(hf_tokenizer, eos_token_ids(hf_model, hf_tokenizer))
    tokenizer.eos_token_ids = []
    return hf_model, tokenizer


def engine(hf_model: Any, tokenizer: Any, use_cuda: bool, **config: Any) -> LLM:
    """An engine over the HF weights, warmed up: the first pass pays kernel setup."""
    llm = LLM(from_hf(hf_model, use_cuda), tokenizer, EngineConfig(**config))
    llm.generate([[1] * 32, [2] * 8], max_tokens=4)
    return llm


def random_prompts(count: int, lengths: Iterable[int], vocab: int, seed: int = 0) -> list[list[int]]:
    """count token-id prompts cycling through lengths, the same on every run."""
    rng = random.Random(seed)
    cycle = list(lengths)
    return [[rng.randrange(vocab) for _ in range(cycle[i % len(cycle)])] for i in range(count)]


def variants() -> list[tuple[str, bool]]:
    """(label, use_cuda) for each engine this machine can run."""
    return [("mini-vllm pure", False)] + ([("mini-vllm cuda", True)] if CUDA_BUILT else [])


def grouped_bars(axes: Axes, groups: list[str], series: dict[str, list[float]],
                 horizontal: bool = False) -> None:
    """One cluster of bars per group, one bar per series in each; a nan value draws no bar."""
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        positions = [group + i * width for group in range(len(groups))]
        if horizontal:
            axes.barh(positions, values, width, label=label)
        else:
            axes.bar(positions, values, width, label=label)
    centers = [group + width * (len(series) - 1) / 2 for group in range(len(groups))]
    if horizontal:
        axes.set_yticks(centers, groups)
    else:
        axes.set_xticks(centers, groups)
        axes.margins(y=0.4)
    axes.legend()


def save_plot(figure: Figure, name: str) -> None:
    """Write figure to results/<name>.png, replacing the previous run's."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {path}")
