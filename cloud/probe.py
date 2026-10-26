"""Checks that an image works on the GPU it landed on: the CUDA extension loads and agrees with
torch, and Triton can JIT a kernel, which needs gcc and Python.h at runtime. Imported only
inside the container: it imports torch and triton at the top."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)


def triton_jit() -> str:
    try:
        x, y = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        out = torch.empty_like(x)
        _add_kernel[(triton.cdiv(1000, 256),)](x, y, out, 1000, BLOCK=256)
        torch.testing.assert_close(out, x + y)
    except Exception as error:  # report, don't raise: the probe's job is to say what works
        return f"failed: {error!r}"
    return "ok"


def extension() -> str:
    try:
        import mini_vllm_ext

        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        got = mini_vllm_ext.rms_norm(x, weight, 1e-6)
        want = torch.nn.functional.rms_norm(x.float(), (128,), eps=1e-6).to(x.dtype) * weight
        error = (got.float() - want.float()).abs().max().item()
    except Exception as failure:
        return f"failed: {failure!r}"
    return f"ok (rms_norm max abs error {error:.2e} against torch)"
