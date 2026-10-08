# Mini-vLLM

A paged-attention LLM inference engine for Qwen3 on NVIDIA GPUs, written in PyTorch and hand-written
CUDA. It serves continuously batched requests out of a paged KV cache, re-forming the batch every
iteration so ragged sequences run in one forward pass, and every CUDA kernel has a pure-PyTorch
counterpart that serves as its test oracle.

The kernels target an RTX 5070 Laptop (Blackwell, sm_120, 8 GB). A KV block is 1.75 MiB in bf16, so
the CLI defaults of 1024 and 2048 blocks fit; 4096 does not. A Qwen3-1.7B target with a Qwen3-0.6B
draft is about 4.6 GB of weights in bf16, and each pool wants about 256 blocks.

## Key features

- **Paged KV cache** of 16-token pages, with copy-on-write sharing for parallel sampling.
- **Radix-tree prefix caching**, so requests behind a shared preamble reuse its pages.
- **Continuous batching** with chunked prefill and preemption.
- **CUDA kernels** for RMSNorm, RoPE, SwiGLU, dense and paged attention, and the cache writes.
  `flash_prefill` is an `mma.sync.m16n8k16` kernel and is the default prefill under `--cuda`.
- **FP8 KV cache**: half the bytes per page, about twice the capacity at the same budget.
- **Speculative decoding**, verified by rejection sampling so the output distribution is unchanged.
  Pass `--draft-model` for a separate draft; if the pair does not fit, use a self-draft instead.

## Usage

```bash
uv sync
uv run python src/extensions/build.py    # the CUDA kernels
```

```python
from mini_vllm import LLM, SamplingParams

llm = LLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_cuda=True,                # needs the CUDA kernels built
    enable_prefix_caching=True,
    fp8_kv_cache=True,
    num_speculative_tokens=4,
)

for completion in llm.generate(["The capital of France is"], max_tokens=32):
    print(completion.text)

for update in llm.generate_stream(["Once upon a time,"], SamplingParams(temperature=0.8), max_tokens=64):
    print(update.text, end="", flush=True)
```

```bash
uv run python main.py "Explain paged attention." --stream --cuda
uv run python batch-main.py --concurrency 32 --cuda --prefix-caching --fp8-kv-cache
uv run pytest
uv run python benches/bench_operators.py
```
