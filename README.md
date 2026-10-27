# Mini-vLLM

A paged-attention inference engine for Qwen3 and Llama 3 in PyTorch, with hand-written CUDA and
Triton kernels. Laid out like [MinivLLM](https://github.com/Wenyueh/MinivLLM): `engine/`,
`layers/`, `models/`, `utils/`.

| feature | where |
|---|---|
| Paged KV cache, copy-on-write, radix prefix cache | `engine/block_manager.py`, `engine/kv_cache.py` |
| Continuous batching, chunked prefill, preemption | `engine/scheduler.py` |
| ModelRunner: KV sizing, decode CUDA graphs, tensor-parallel workers | `engine/model_runner.py` |
| Speculative decoding (self-draft or draft model) | `engine/speculative.py` |
| Backends: `torch` (oracle), `cuda` (`src/extensions/`), `triton` | `layers/backends/` |
| fp8 KV pages, parallel sampling (n > 1) | `layers/attention.py`, `engine/llm_engine.py` |
| Safetensors loader with packed q/k/v and gate/up | `utils/loader.py` |

```bash
uv sync
uv run python src/extensions/build.py          # CUDA kernels for this GPU (MINI_VLLM_CUDA_ARCHS to override)
uv run python main.py "Explain paged attention." --backend cuda --stream
uv run pytest                                  # GPU tests skip on a machine without one
```

```python
from mini_vllm import LLM

llm = LLM.from_pretrained("Qwen/Qwen3-8B", backend="triton", enable_prefix_caching=True)
print(llm.generate(["The capital of France is"], max_tokens=32)[0].text)
```

`tensor_parallel_size=2` splits the model over two GPUs; `num_speculative_tokens=4` drafts with
the model's own first layers.

## H100s on Modal

```bash
uv sync --group cloud
uv run --group cloud modal run -m cloud.app::download            # models into a volume, once
uv run --group cloud modal run -m cloud.app::tests               # tests on an H100 (--gpus 2 for TP)
uv run --group cloud modal run -m cloud.app::sweep --suite quick --engines mini-cuda,mini-triton,vllm
```

Results: [docs/perf/baseline.md](docs/perf/baseline.md) (before the refactor) and
[docs/perf/final.md](docs/perf/final.md).
