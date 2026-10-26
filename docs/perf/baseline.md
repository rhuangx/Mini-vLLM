# Baseline: Qwen3-8B on one H100, before the refactor

Every later phase of the refactor is measured against these numbers. The engine code is that of
`028765a`. Phase 0 added only instrumentation and the measurement harness, and the run below was
made at `a20dd43`.

- **Hardware:** Modal `gpu="H100!"`, an NVIDIA H100 80GB HBM3 (SXM, 3.35 TB/s), driver 610.57.04.
- **Software:** Mini-vLLM on torch 2.14.1+cu130 with its CUDA extension built for sm_90a; vLLM 0.31.0
  on torch 2.13.0+cu130.
- **Correctness:** the full test suite passes on the H100, 320 passed with none skipped, including
  every CUDA-kernel and real-Qwen3 test.

## The four engines

Each engine runs on its own H100 under the same limits:

- 12,288 KV blocks of 16 tokens (196k tokens; vLLM gets exactly this through
  `num_gpu_blocks_override`)
- at most 256 sequences and 4,096 batched tokens, `max_model_len` 4,096
- prefix caching on, bf16 weights, greedy sampling with EOS ignored

| engine | what it is |
|---|---|
| Mini-vLLM, CUDA kernels | Mini-vLLM with its hand-written kernels for RMSNorm, RoPE, SwiGLU, paged attention and prefill. No CUDA graphs and no `torch.compile`. |
| Mini-vLLM, pure PyTorch | The same engine on its pure-PyTorch path, the oracle its kernels are tested against. It is measured only at 1 and 16 in flight, being about ten times slower. |
| vLLM 0.31.0 | vLLM as shipped, with CUDA graphs and `torch.compile`: production speed. |
| vLLM 0.31.0, eager | vLLM with `enforce_eager=True`, so no CUDA graphs and no compilation. This is the fair comparison for Mini-vLLM today, since Mini-vLLM has neither yet. |

vLLM's FlashInfer sampler is off because the image has no nvcc, and greedy requests never reach that
sampler.

## The workloads

The prompts are random token ids, seeded by each trace's parameters, so every engine and every run
sees the same requests.

| workload | shape | load |
|---|---|---|
| decode-heavy | 128-token prompts, 512-token outputs | 1, 2, 4 … 256 requests in flight; a new one starts as one finishes |
| prefill-heavy | 2,048-token prompts, 16-token outputs | 8 in flight |
| mixed | lognormal lengths around 512 in and 128 out | Poisson arrivals at 1, 2, 4, 8 and 16 requests/s |
| shared-prefix | a 1,024-token preamble shared by every prompt, plus 64 unique tokens; 64-token outputs | 8 in flight |

One loop drives every engine (`cloud/workload.py`), and a token is timestamped when the step that
made it returns.

- **Time to first token:** counted from arrival, so it includes queueing.
- **Time per output token:** (last token − first) / (n − 1), per request.
- **Decode memory-bandwidth utilization:** (15.1 GB of weights + 144 KiB × context tokens) / step
  time / 3.35 TB/s.

Reproduce it from the repository root:

```bash
uv run --group cloud modal run -m cloud.app::sweep --suite full --engines mini-cuda,mini-torch,vllm,vllm-eager --tag baseline
```

![Decode-heavy: output tokens/s and time per output token against requests in flight](baseline-decode-heavy.png)

## What the numbers say

1. **At low concurrency, Mini-vLLM waits on kernel launches.**
   - With 16 requests in flight, a decode step takes 20.3 ms. The GPU runs kernels for 8.9 ms of
     it (44% busy), across 884 launches.
   - The kernels themselves are fine: the GEMMs read 15.1 GB of weights in 6.3 ms, about 2.4 TB/s
     or 72% of peak.
   - Time per output token at one request is 11.4 ms, against 6.7 ms for vLLM and 17.8 ms for
     vLLM eager.
   - Next: CUDA graphs (Phase 8).
2. **At high concurrency, attention is the bottleneck.**
   - With 256 in flight, the GPU is 90% busy, and the paged attention kernel takes 29.1 of the
     38.1 ms.
   - It runs one 1,024-thread block per (token, query head), so every K/V tile is read once per
     head in the GQA group of 4.
   - Throughput is 3.8k tokens/s, against 16.3k for vLLM and 11.3k for vLLM eager.
   - Memory-bandwidth utilization falls from 41% at 4 in flight to 14% at 256.
   - Next: a GQA-packed Triton kernel (Phase 7), then split-K (Phase 9).
3. **Long prefills stall decodes.**
   - In the prefill-heavy workload, time to first token at p99 is 1,151 ms against 395 ms.
   - Decodes that share a step with a 2,048-token prefill take 67 ms per token against 16 ms.
   - Long chunks gather their pages and run `flash_prefill` one sequence at a time, from Python.
   - Next: decode-first slicing (Phase 6) and a paged prefill kernel (Phase 7).
4. **The prefix cache misses requests that run at the same time.**
   - In the shared-prefix workload, time to first token at p50 is 120 ms against 30 ms.
   - Mini-vLLM adds pages to the radix cache only when a request is freed, so the 8 requests in
     flight at once each prefill the whole 1,024-token preamble.
5. **The pure-PyTorch path is the oracle, not a contender.**
   - Time per output token is 29 ms at one request and 132 ms at 16.
   - It launches about 16k kernels a step, and the GPU is 15–24% busy.

## How far one run can be trusted

The same sweep was run twice, on two Modal containers. Where the GPU is the bottleneck, the two
runs agreed; where Python overhead is the bottleneck, they did not. The tables compare only the
workloads whose traces were identical in both runs.

| engine | change in throughput between the two runs |
|---|---|
| vLLM 0.31.0 | within 1.5% everywhere |
| Mini-vLLM, CUDA kernels | within 1% at 128 and 256 in flight, prefill-heavy and shared-prefix; −5% to +15% at 1–64 in flight |
| vLLM 0.31.0, eager | −3% to −35% |
| Mini-vLLM, pure PyTorch | +62% |

The host CPU sets the pace of a Python-bound step, and containers land on different hosts. Every
run now records the host CPU model; this baseline predates that. So a ±3% regression check is
only meaningful for GPU-bound points. Host-bound points need both sides measured in the same
container, or the median of repeated runs.

## Full tables

Generated by `cloud/report.py` from [baseline.json](baseline.json).

### Environment

| engine | device | host CPU | driver | torch | CUDA | load s | KV blocks |
|---|---|---|---|---|---|---|---|
| Mini-vLLM, CUDA kernels | NVIDIA H100 80GB HBM3 | not recorded | 610.57.04 | 2.14.1+cu130 | 13.0 | 10.2 | 12288 |
| Mini-vLLM, pure PyTorch | NVIDIA H100 80GB HBM3 | not recorded | 610.57.04 | 2.14.1+cu130 | 13.0 | 8.4 | 12288 |
| vLLM 0.31.0 | NVIDIA H100 80GB HBM3 | not recorded | 610.57.04 | 2.13.0+cu130 | 13.0 | 112.2 | 12288 |
| vLLM 0.31.0, eager | NVIDIA H100 80GB HBM3 | not recorded | 610.57.04 | 2.13.0+cu130 | 13.0 | 45.9 | 12288 |

### Decode-heavy: 128-token prompts, 512-token outputs, a fixed number in flight

**Output tokens/s**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 87 | 35 | 148 | 56 |
| 2 in flight | 174 | - | 295 | 112 |
| 4 in flight | 351 | - | 583 | 169 |
| 8 in flight | 695 | - | 1,140 | 439 |
| 16 in flight | 1,304 | 121 | 2,205 | 844 |
| 32 in flight | 2,027 | - | 4,117 | 1,788 |
| 64 in flight | 2,648 | - | 7,255 | 3,395 |
| 128 in flight | 3,383 | - | 11,691 | 6,625 |
| 256 in flight | 3,775 | - | 16,251 | 11,270 |

**Time to first token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 15.0 | 28.6 | 11.7 | 39.1 |
| 2 in flight | 27.6 | - | 13.7 | 49.7 |
| 4 in flight | 37.5 | - | 22.0 | 73.7 |
| 8 in flight | 64.9 | - | 39.1 | 59.1 |
| 16 in flight | 117.5 | 155.0 | 57.9 | 78.6 |
| 32 in flight | 231.0 | - | 126.6 | 134.0 |
| 64 in flight | 264.2 | - | 192.6 | 224.4 |
| 128 in flight | 264.2 | - | 284.8 | 282.1 |
| 256 in flight | 468.5 | - | 308.5 | 304.8 |

**Time to first token, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 21.9 | 35.1 | 12.4 | 46.8 |
| 2 in flight | 27.6 | - | 13.7 | 67.7 |
| 4 in flight | 37.5 | - | 24.6 | 89.0 |
| 8 in flight | 64.9 | - | 42.1 | 78.6 |
| 16 in flight | 117.5 | 155.0 | 61.2 | 110.1 |
| 32 in flight | 231.0 | - | 131.5 | 151.0 |
| 64 in flight | 501.2 | - | 210.7 | 261.1 |
| 128 in flight | 1,005.4 | - | 411.3 | 430.8 |
| 256 in flight | 2,058.8 | - | 833.2 | 883.2 |

**Time per output token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 11.42 | 28.54 | 6.74 | 17.77 |
| 2 in flight | 11.47 | - | 6.76 | 17.77 |
| 4 in flight | 11.45 | - | 6.83 | 23.97 |
| 8 in flight | 11.41 | - | 6.95 | 18.20 |
| 16 in flight | 12.08 | 132.09 | 7.16 | 19.29 |
| 32 in flight | 15.39 | - | 7.56 | 17.73 |
| 64 in flight | 23.77 | - | 8.45 | 18.52 |
| 128 in flight | 37.26 | - | 10.37 | 19.03 |
| 256 in flight | 66.87 | - | 15.10 | 21.92 |

**Inter-token latency, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 11.57 | 29.58 | 7.09 | 20.71 |
| 2 in flight | 11.59 | - | 7.12 | 23.05 |
| 4 in flight | 11.59 | - | 7.31 | 46.49 |
| 8 in flight | 11.52 | - | 7.28 | 21.31 |
| 16 in flight | 13.25 | 136.83 | 7.63 | 22.46 |
| 32 in flight | 18.35 | - | 8.22 | 19.95 |
| 64 in flight | 35.14 | - | 9.50 | 23.05 |
| 128 in flight | 47.38 | - | 23.49 | 21.77 |
| 256 in flight | 236.86 | - | 34.27 | 29.94 |

**Decode memory-bandwidth utilization %, estimated from ITL**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 39.7 | 16.0 | 67.3 | 25.1 |
| 2 in flight | 39.7 | - | 67.3 | 25.4 |
| 4 in flight | 40.7 | - | 67.2 | 20.1 |
| 8 in flight | 40.8 | - | 66.8 | 25.4 |
| 16 in flight | 40.3 | 3.6 | 67.0 | 25.4 |
| 32 in flight | 33.0 | - | 66.9 | 27.7 |
| 64 in flight | 24.1 | - | 66.6 | 30.2 |
| 128 in flight | 18.4 | - | 67.5 | 36.0 |
| 256 in flight | 13.6 | - | 62.0 | 40.8 |

**Decode memory-bandwidth utilization %, Mini-vLLM's own forward GPU time (p50)**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 41.2 | 16.2 | - | - |
| 2 in flight | 41.3 | - | - | - |
| 4 in flight | 42.3 | - | - | - |
| 8 in flight | 42.9 | - | - | - |
| 16 in flight | 42.4 | 3.7 | - | - |
| 32 in flight | 34.9 | - | - | - |
| 64 in flight | 25.6 | - | - | - |
| 128 in flight | 19.7 | - | - | - |
| 256 in flight | 14.6 | - | - | - |

**Decode memory-bandwidth utilization %, Mini-vLLM's whole step (p50)**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 in flight | 39.8 | 16.0 | - | - |
| 2 in flight | 39.8 | - | - | - |
| 4 in flight | 40.6 | - | - | - |
| 8 in flight | 40.9 | - | - | - |
| 16 in flight | 40.3 | 3.6 | - | - |
| 32 in flight | 33.1 | - | - | - |
| 64 in flight | 24.3 | - | - | - |
| 128 in flight | 18.6 | - | - | - |
| 256 in flight | 13.8 | - | - | - |

### Prefill-heavy: 2048-token prompts, 16-token outputs, a fixed number in flight

**Output tokens/s**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 96 | - | 259 | 202 |

**Time to first token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 275.2 | - | 242.8 | 248.9 |

**Time to first token, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 1,150.8 | - | 394.6 | 407.0 |

**Time per output token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 67.34 | - | 16.40 | 26.34 |

**Inter-token latency, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 275.18 | - | 97.95 | 118.37 |

### Mixed: lognormal lengths around 512 in and 128 out, Poisson arrivals

**Output tokens/s**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 req/s | 110 | - | 113 | 106 |
| 2 req/s | 295 | - | 322 | 267 |
| 4 req/s | 509 | - | 548 | 475 |
| 8 req/s | 905 | - | 1,048 | 902 |
| 16 req/s | 1,139 | - | 1,957 | 1,403 |

**Time to first token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 req/s | 42.4 | - | 29.6 | 63.4 |
| 2 req/s | 46.9 | - | 32.7 | 64.3 |
| 4 req/s | 45.9 | - | 32.9 | 64.3 |
| 8 req/s | 59.7 | - | 35.4 | 63.2 |
| 16 req/s | 157.2 | - | 42.1 | 64.2 |

**Time to first token, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 req/s | 158.3 | - | 43.3 | 78.4 |
| 2 req/s | 107.9 | - | 60.9 | 79.8 |
| 4 req/s | 203.6 | - | 55.9 | 83.0 |
| 8 req/s | 160.7 | - | 68.0 | 82.0 |
| 16 req/s | 412.8 | - | 157.0 | 119.1 |

**Time per output token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 req/s | 11.71 | - | 6.80 | 17.56 |
| 2 req/s | 12.28 | - | 6.98 | 17.47 |
| 4 req/s | 13.70 | - | 7.20 | 17.82 |
| 8 req/s | 21.63 | - | 7.84 | 17.51 |
| 16 req/s | 51.96 | - | 9.73 | 18.11 |

**Inter-token latency, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 1 req/s | 14.01 | - | 7.50 | 19.70 |
| 2 req/s | 39.81 | - | 10.28 | 19.72 |
| 4 req/s | 49.38 | - | 20.64 | 21.79 |
| 8 req/s | 88.66 | - | 22.55 | 23.02 |
| 16 req/s | 203.66 | - | 27.91 | 37.31 |

### Shared-prefix: a 1024-token preamble shared by every prompt plus 64 unique tokens, 64-token outputs, a fixed number in flight

**Output tokens/s**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 488 | - | 1,072 | 442 |

**Time to first token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 120.0 | - | 30.2 | 60.6 |

**Time to first token, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 490.7 | - | 45.7 | 64.4 |

**Time per output token, p50 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 13.99 | - | 7.06 | 17.39 |

**Inter-token latency, p99 ms**

| load | Mini-vLLM, CUDA kernels | Mini-vLLM, pure PyTorch | vLLM 0.31.0 | vLLM 0.31.0, eager |
|---|---|---|---|---|
| 8 in flight | 93.44 | - | 7.46 | 20.65 |

### Profile: Mini-vLLM, CUDA kernels, decode 16 in flight

A step takes 20.33 ms, of which the GPU runs kernels for 8.90 ms (44% busy); 884 kernels per step, over 10 steps.

| GPU time by category | ms/step |
|---|---|
| gemm | 6.258 |
| attention | 1.877 |
| norm | 0.309 |
| rope | 0.127 |
| elementwise | 0.118 |
| kv_write | 0.095 |
| activation | 0.080 |
| copy | 0.022 |
| sampling | 0.011 |

| host span | ms/step |
|---|---|
| schedule | 0.075 |
| prepare | 0.314 |
| forward | 19.211 |
| sample | 0.504 |
| spec.verify | 0.014 |
| commit | 0.024 |

| top kernel | category | ms/step | calls/step |
|---|---|---|---|
| `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | gemm | 2.666 | 72 |
| `void (anonymous namespace)::paged_attention_kernel<c10::BFloat16, c10::BFloat16>` | attention | 1.877 | 36 |
| `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | gemm | 1.369 | 36 |
| `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | gemm | 1.024 | 72 |
| `nvjet_sm90_tst_384x16_64x4_2x1_v_bz_TNT` | gemm | 0.441 | 1 |
| `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | gemm | 0.427 | 72 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __n` | gemm | 0.331 | 180 |
| `void (anonymous namespace)::rms_norm_kernel<c10::BFloat16, mini_vllm::Vector<c10` | norm | 0.309 | 145 |
| `void (anonymous namespace)::rope_kernel<c10::BFloat16>(c10::BFloat16 const*, int` | rope | 0.127 | 72 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c1` | elementwise | 0.118 | 72 |

### Profile: Mini-vLLM, CUDA kernels, decode 256 in flight

A step takes 42.41 ms, of which the GPU runs kernels for 38.10 ms (90% busy); 704 kernels per step, over 10 steps.

| GPU time by category | ms/step |
|---|---|
| attention | 29.131 |
| gemm | 7.499 |
| norm | 0.558 |
| rope | 0.231 |
| elementwise | 0.208 |
| activation | 0.205 |
| kv_write | 0.140 |
| copy | 0.074 |
| sampling | 0.056 |

| host span | ms/step |
|---|---|
| schedule | 0.990 |
| prepare | 1.172 |
| forward | 17.433 |
| sample | 22.175 |
| spec.verify | 0.023 |
| commit | 0.145 |

| top kernel | category | ms/step | calls/step |
|---|---|---|---|
| `void (anonymous namespace)::paged_attention_kernel<c10::BFloat16, c10::BFloat16>` | attention | 29.131 | 36 |
| `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | gemm | 3.199 | 108 |
| `nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT` | gemm | 3.063 | 72 |
| `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` | gemm | 0.748 | 72 |
| `void (anonymous namespace)::rms_norm_kernel<c10::BFloat16, mini_vllm::Vector<c10` | norm | 0.558 | 145 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | gemm | 0.490 | 1 |
| `void (anonymous namespace)::rope_kernel<c10::BFloat16>(c10::BFloat16 const*, int` | rope | 0.231 | 72 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c1` | elementwise | 0.208 | 72 |
| `void (anonymous namespace)::swiglu_kernel<c10::BFloat16, mini_vllm::Vector<c10::` | activation | 0.205 | 36 |
| `void (anonymous namespace)::paged_cache_update_kernel<c10::BFloat16>(int const*,` | kv_write | 0.140 | 72 |

### Profile: Mini-vLLM, CUDA kernels, mixed

A step takes 29.18 ms, of which the GPU runs kernels for 19.28 ms (66% busy); 1,013 kernels per step, over 10 steps.

| GPU time by category | ms/step |
|---|---|
| attention | 9.494 |
| gemm | 7.891 |
| copy | 0.627 |
| norm | 0.519 |
| rope | 0.224 |
| activation | 0.193 |
| elementwise | 0.185 |
| kv_write | 0.136 |
| sampling | 0.011 |

| host span | ms/step |
|---|---|
| schedule | 0.089 |
| prepare | 0.488 |
| forward | 27.701 |
| sample | 0.614 |
| spec.verify | 0.015 |
| commit | 0.078 |

| top kernel | category | ms/step | calls/step |
|---|---|---|---|
| `(anonymous namespace)::flash_prefill_kernel(__nv_bfloat16 const*, __nv_bfloat16 ` | attention | 4.819 | 14.4 |
| `void (anonymous namespace)::paged_attention_kernel<c10::BFloat16, c10::BFloat16>` | attention | 4.675 | 36 |
| `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | gemm | 1.636 | 43.2 |
| `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | gemm | 0.821 | 21.6 |
| `nvjet_sm90_tst_192x176_64x4_1x2_h_bz_coopB_TNT` | gemm | 0.656 | 7.2 |
| `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT` | gemm | 0.609 | 43.2 |
| `nvjet_sm90_tst_192x144_64x5_1x2_h_bz_coopB_TNT` | gemm | 0.549 | 7.2 |
| `nvjet_sm90_tst_128x176_64x5_2x1_v_bz_TNT` | gemm | 0.534 | 10.8 |
| `void (anonymous namespace)::rms_norm_kernel<c10::BFloat16, mini_vllm::Vector<c10` | norm | 0.519 | 145 |
| `nvjet_sm90_tst_128x144_64x6_2x1_v_bz_TNT` | gemm | 0.451 | 10.8 |

### Profile: Mini-vLLM, pure PyTorch, decode 16 in flight

A step takes 271.30 ms, of which the GPU runs kernels for 40.43 ms (15% busy); 15,975 kernels per step, over 10 steps.

| GPU time by category | ms/step |
|---|---|
| copy | 17.222 |
| elementwise | 11.522 |
| gemm | 10.375 |
| other | 1.094 |
| kv_write | 0.205 |
| sampling | 0.011 |

| host span | ms/step |
|---|---|
| schedule | 0.064 |
| prepare | 0.251 |
| forward | 270.312 |
| sample | 0.461 |
| spec.verify | 0.011 |
| commit | 0.022 |

| top kernel | category | ms/step | calls/step |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a` | copy | 5.389 | 1152 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda` | copy | 5.332 | 1945 |
| `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | gemm | 2.729 | 72 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, lo` | copy | 2.318 | 1299 |
| `void gemvNSP_kernel<float, float, float, float, 1, 16, 4, 1024, false, cublasGem` | gemm | 2.096 | 576 |
| `void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGem` | gemm | 1.888 | 576 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a` | elementwise | 1.849 | 1152 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f` | elementwise | 1.618 | 576 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f` | elementwise | 1.517 | 576 |
| `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_splitK_TNT` | gemm | 1.375 | 36 |

### Profile: Mini-vLLM, pure PyTorch, mixed

A step takes 260.23 ms, of which the GPU runs kernels for 62.30 ms (24% busy); 15,903 kernels per step, over 10 steps.

| GPU time by category | ms/step |
|---|---|
| copy | 30.160 |
| elementwise | 16.002 |
| gemm | 14.742 |
| other | 1.150 |
| kv_write | 0.232 |
| sampling | 0.011 |

| host span | ms/step |
|---|---|
| schedule | 0.081 |
| prepare | 0.457 |
| forward | 258.881 |
| sample | 0.498 |
| spec.verify | 0.012 |
| commit | 0.080 |

| top kernel | category | ms/step | calls/step |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a` | copy | 15.817 | 1152 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda` | copy | 6.555 | 1945 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, lo` | copy | 2.973 | 1299 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a` | elementwise | 2.659 | 1152 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f` | elementwise | 2.286 | 576 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f` | elementwise | 2.232 | 576 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, fl` | gemm | 1.834 | 342 |
| `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` | gemm | 1.681 | 43.2 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, fl` | gemm | 1.604 | 457.2 |
| `sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_` | gemm | 1.532 | 18 |
