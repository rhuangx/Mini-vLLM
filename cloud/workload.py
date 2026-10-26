"""Engine-agnostic workloads: seeded request traces, and the loop that drives an engine through
one while recording when each request arrived, got each token and finished.

Imports neither engine, so one driver runs Mini-vLLM and vLLM through the same Adapter: what
differs between them is the engine, never the measurement.
"""

from __future__ import annotations

import math
import random
import time
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol

# Random prompts draw from ordinary tokens: Qwen3's special tokens start at 151643.
TOKEN_LOW, TOKEN_HIGH = 1000, 150_000


@dataclass(frozen=True)
class Workload:
    """A request trace. Closed loop keeps `concurrency` requests in flight, starting the next as
    one finishes; Poisson submits at `rate` requests per second whatever the engine keeps up with.
    With lognormal, input_len and output_len are medians, not exact lengths."""

    name: str
    num_requests: int
    input_len: int
    output_len: int
    concurrency: int = 0      # closed loop when > 0
    rate: float = 0.0         # Poisson arrivals when > 0
    lognormal: bool = False
    shared_prefix: int = 0    # tokens every prompt starts with, for the prefix cache
    seed: int = 0

    def requests(self) -> list[RequestSpec]:
        """The trace, identical on every run and for every engine."""
        rng = random.Random(zlib.crc32(self.name.encode()) + self.seed)
        prefix = [rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(self.shared_prefix)]
        specs, arrival = [], 0.0
        for _ in range(self.num_requests):
            input_len, output_len = self.input_len, self.output_len
            if self.lognormal:
                input_len = _clip(rng.lognormvariate(math.log(self.input_len), 0.6), 32, 2048)
                output_len = _clip(rng.lognormvariate(math.log(self.output_len), 0.6), 16, 512)
            if self.rate > 0:
                arrival += rng.expovariate(self.rate)
            prompt = prefix + [rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(input_len)]
            specs.append(RequestSpec(prompt, output_len, arrival))
        return specs


def _clip(value: float, low: int, high: int) -> int:
    return max(low, min(high, round(value)))


@dataclass(frozen=True)
class RequestSpec:
    prompt: list[int]
    max_tokens: int
    arrival: float  # seconds after the start for Poisson; closed loop submits when a slot frees


@dataclass
class RequestRecord:
    """One request's timeline, in seconds from the start of the workload."""

    input_len: int
    max_tokens: int
    arrival: float = 0.0
    token_times: list[float] = field(default_factory=list)  # one per token; a run of k shares a time
    finish: float | None = None

    def to_json(self) -> dict:
        record = asdict(self)
        record["token_times"] = [round(t, 5) for t in self.token_times]
        return record


class Adapter(Protocol):
    def add(self, key: int, prompt: list[int], max_tokens: int) -> None: ...
    def step(self) -> list[tuple[int, int, bool]]: ...  # (key, new tokens, finished) per request
    def has_work(self) -> bool: ...


def run(
    adapter: Adapter, workload: Workload, after_step: Callable[[], None] | None = None
) -> tuple[list[RequestRecord], float]:
    """Drive the engine through the workload; returns each request's record and the wall time.

    Requests are submitted between steps, which is when an engine stepped from one loop can
    accept them; a Poisson request that came due mid-step counts its wait from when it was due.
    """
    specs = workload.requests()
    records = [RequestRecord(len(spec.prompt), spec.max_tokens) for spec in specs]
    start = time.perf_counter()
    submitted = in_flight = finished = 0
    while finished < len(specs):
        now = time.perf_counter() - start
        if workload.concurrency > 0:
            while in_flight < workload.concurrency and submitted < len(specs):
                records[submitted].arrival = now
                adapter.add(submitted, specs[submitted].prompt, specs[submitted].max_tokens)
                submitted += 1
                in_flight += 1
        else:
            while submitted < len(specs) and specs[submitted].arrival <= now:
                records[submitted].arrival = specs[submitted].arrival
                adapter.add(submitted, specs[submitted].prompt, specs[submitted].max_tokens)
                submitted += 1
                in_flight += 1
            if in_flight == 0:
                time.sleep(max(0.0, specs[submitted].arrival - now))
                continue

        events = adapter.step()
        now = time.perf_counter() - start
        for key, count, done in events:
            records[key].token_times.extend([now] * count)
            if done:
                records[key].finish = now
                in_flight -= 1
                finished += 1
        if after_step is not None:
            after_step()
    wall = max(record.finish for record in records) - min(record.arrival for record in records)
    return records, wall


def w1(concurrency: int) -> Workload:
    """Decode-heavy: short prompts, long outputs."""
    return Workload(f"W1-c{concurrency}", max(4, 2 * concurrency), 128, 512, concurrency=concurrency)


W2 = Workload("W2", 32, 2048, 16, concurrency=8)  # prefill-heavy


def w3(rate: float) -> Workload:
    """Mixed: lognormal lengths around 512 in and 128 out, Poisson arrivals."""
    return Workload(f"W3-r{rate:g}", min(128, max(32, int(rate * 20))), 512, 128, rate=rate, lognormal=True)


W4 = Workload("W4", 32, 64, 64, concurrency=8, shared_prefix=1024)  # 1024 shared + 64 unique

SUITES: dict[str, list[Workload]] = {
    "quick": [*(w1(c) for c in (1, 16, 64, 256)), W2],
    "full": [
        *(w1(c) for c in (1, 2, 4, 8, 16, 32, 64, 128, 256)),
        W2,
        *(w3(r) for r in (1, 2, 4, 8, 16)),
        W4,
    ],
    # The pure-torch backend loops over sequences in Python: two points are its baseline.
    "torch": [w1(1), w1(16)],
    "smoke": [Workload("smoke", 4, 64, 16, concurrency=2)],
    "none": [],  # profiles only
}

# Traced, not timed, from step 20 on: decode-only windows at 16 and 256 in flight, and a window
# where short outputs keep requests finishing, so most steps also prefill a newly admitted prompt.
PROFILES: dict[str, Workload] = {
    "decode": Workload("profile-decode", 16, 128, 64, concurrency=16),
    "decode-c256": Workload("profile-decode-c256", 256, 128, 96, concurrency=256),
    "mixed": Workload("profile-mixed", 96, 512, 24, concurrency=16, lognormal=True),
}
