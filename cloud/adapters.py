"""Both engines behind workload.Adapter: add a token prompt, step, report new tokens. Greedy,
ignoring EOS, no detokenizing, so each request costs exactly its input and output tokens.

Each engine is imported inside its own functions: the Mini-vLLM image has no vLLM, and the
vLLM image has no Mini-vLLM.
"""

from __future__ import annotations

import gc
import inspect
import time
from collections import Counter
from typing import Any

# One engine configuration for both, so the comparison is engine against engine: same pages,
# same page size, same batch limits.
ENGINE = {
    "model": "Qwen/Qwen3-8B",
    "block_size": 16,
    "num_blocks": 12288,          # 196k tokens of KV: W1 at 256 in flight needs about 164k
    "max_num_seqs": 256,
    "max_num_batched_tokens": 4096,
    "max_model_len": 4096,
    "prefix_caching": True,
}


class MiniVllmAdapter:
    """One Mini-vLLM LLM. steps keeps (wall seconds, StepStats) for each step()."""

    def __init__(self, llm: Any) -> None:
        from mini_vllm import RequestStatus

        self.llm = llm
        self.finished_status = RequestStatus.FINISHED
        self.keys: dict[int, int] = {}
        self.steps: list[tuple[float, Any]] = []

    def add(self, key: int, prompt: list[int], max_tokens: int) -> None:
        request = self.llm.add_request(prompt, max_tokens=max_tokens, ignore_eos=True)
        self.keys[request.request_id] = key

    def has_work(self) -> bool:
        return self.llm.scheduler.has_work

    def step(self) -> list[tuple[int, int, bool]]:
        began = time.perf_counter()
        emitted, _ = self.llm.step()
        self.steps.append((time.perf_counter() - began, getattr(self.llm, "last_step", None)))
        counts: Counter[int] = Counter()
        done: set[int] = set()
        for request, _ in emitted:
            counts[request.request_id] += 1
            if request.status is self.finished_status:
                done.add(request.request_id)
        return [(self.keys[rid], count, rid in done) for rid, count in counts.items()]


class MiniVllmFactory:
    """The model loads once; every workload gets a fresh engine over it, so no KV pool, prefix
    cache or scheduler state carries from one workload into the next."""

    def __init__(self, backend: str, engine: dict = ENGINE) -> None:
        import torch

        from mini_vllm import LLM, EngineConfig

        self.torch, self.LLM = torch, LLM
        self.config = EngineConfig(
            num_blocks=engine["num_blocks"],
            block_size=engine["block_size"],
            max_sequences=engine["max_num_seqs"],
            max_batched_tokens=engine["max_num_batched_tokens"],
            chunk_size=engine["max_num_batched_tokens"],  # like vLLM: one request may take the budget
            enable_prefix_caching=engine["prefix_caching"],
        )
        # Phase 2 replaces use_cuda with backend; take whichever this checkout has.
        parameters = inspect.signature(LLM.from_pretrained).parameters
        if "backend" in parameters:
            kernel = {"backend": backend}
        elif backend in ("torch", "cuda"):
            kernel = {"use_cuda": backend == "cuda"}
        else:
            raise ValueError(f"backend {backend!r} needs a checkout with the backend switch")
        began = time.perf_counter()
        base = LLM.from_pretrained(engine["model"], **kernel, num_blocks=1)
        self.load_s = time.perf_counter() - began
        self.model, self.tokenizer = base.model, base.tokenizer
        del base
        self._release()

    def _release(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()

    def fresh(self) -> MiniVllmAdapter:
        llm = self.LLM(self.model, self.tokenizer, self.config)
        llm.generate([[1000] * 32, [1001] * 8], max_tokens=4)  # the first pass pays kernel setup
        return MiniVllmAdapter(llm)

    def release(self, adapter: MiniVllmAdapter) -> None:
        adapter.llm = None
        self._release()


class VllmAdapter:
    """vLLM's V1 LLMEngine, stepped from the same loop; request ids are unique per workload."""

    def __init__(self, engine: Any) -> None:
        from vllm import SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        self.engine = engine
        self.SamplingParams, self.TokensPrompt = SamplingParams, TokensPrompt
        self.delta = RequestOutputKind.DELTA
        self.prefix = f"w{time.monotonic_ns()}-"

    def add(self, key: int, prompt: list[int], max_tokens: int) -> None:
        params = self.SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True, output_kind=self.delta, detokenize=False
        )
        self.engine.add_request(f"{self.prefix}{key}", self.TokensPrompt(prompt_token_ids=prompt), params)

    def has_work(self) -> bool:
        return self.engine.has_unfinished_requests()

    def step(self) -> list[tuple[int, int, bool]]:
        return [
            (int(out.request_id.removeprefix(self.prefix)), len(out.outputs[0].token_ids), out.finished)
            for out in self.engine.step()
        ]


def vllm_engine(enforce_eager: bool, engine: dict = ENGINE) -> tuple[Any, float]:
    """A vLLM engine with Mini-vLLM's limits and exactly its KV capacity; returns it and its load time."""
    from vllm import EngineArgs, LLMEngine

    began = time.perf_counter()
    llm_engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=engine["model"],
            dtype="bfloat16",
            block_size=engine["block_size"],
            num_gpu_blocks_override=engine["num_blocks"],
            max_num_seqs=engine["max_num_seqs"],
            max_num_batched_tokens=engine["max_num_batched_tokens"],
            max_model_len=engine["max_model_len"],
            enable_prefix_caching=engine["prefix_caching"],
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.9,
            seed=0,
        )
    )
    adapter = VllmAdapter(llm_engine)
    for key in range(2):  # warm up, as the Mini-vLLM factory does
        adapter.add(key, [1000 + key] * 32, 4)
    while adapter.has_work():
        adapter.step()
    return llm_engine, time.perf_counter() - began
