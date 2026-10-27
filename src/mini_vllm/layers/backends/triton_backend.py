"""The Triton backend: the kernels in triton_kernels.py, launched with fixed configs. Dense
attention, which serving never takes, stays on the pure path."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import triton

from mini_vllm.layers.backends import SHORT_QUERY
from mini_vllm.layers.backends import triton_kernels as kernels
from mini_vllm.layers.backends.cuda_backend import split_short
from mini_vllm.layers.backends.torch_backend import TorchBackend

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import HostLists

__all__ = ["TritonBackend"]


def _rows(x: torch.Tensor) -> torch.Tensor:
    """x as contiguous rows over its last axis."""
    return x.contiguous().view(-1, x.shape[-1])


class TritonBackend(TorchBackend):
    name = "triton"
    supports_graphs = True

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("the triton backend needs a CUDA GPU")

    def _norm(self, x, residual, weight, eps):
        rows = _rows(x)
        out = torch.empty_like(rows)
        summed = torch.empty_like(rows) if residual is not None else out
        res_rows = _rows(residual) if residual is not None else rows
        kernels.rms_norm_kernel[(rows.shape[0],)](
            rows, res_rows, weight, out, summed, rows.stride(0), res_rows.stride(0), rows.shape[1], eps,
            HAS_RESIDUAL=residual is not None, BLOCK=triton.next_power_of_2(rows.shape[1]),
        )
        return out.view(x.shape), summed.view(x.shape)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return self._norm(x, None, weight, eps)[0]

    def add_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._norm(x, residual, weight, eps)

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        rows = _rows(x)
        half = rows.shape[1] // 2
        out = torch.empty(rows.shape[0], half, dtype=x.dtype, device=x.device)
        block = 1024
        kernels.silu_and_mul_kernel[(rows.shape[0], triton.cdiv(half, block))](
            rows, out, rows.stride(0), half, BLOCK=block
        )
        return out.view(*x.shape[:-1], half)

    def rope(
        self, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, length, heads, dim = x.shape
        flat = x.reshape(batch * length, heads, dim)
        out = torch.empty(batch * length, heads, dim, dtype=x.dtype, device=x.device)
        kernels.rope_kernel[(batch * length, heads)](
            flat, positions.reshape(-1), cos, sin, out, heads, flat.stride(0), flat.stride(1),
            HALF=dim // 2, BLOCK=triton.next_power_of_2(dim // 2),
        )
        return out.view(x.shape)

    def write_kv(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fp8_scales: tuple[float, float] | None = None,
    ) -> None:
        row = key.shape[-2] * key.shape[-1]
        scales = fp8_scales or (1.0, 1.0)
        kernels.store_kv_kernel[(key.shape[0],)](
            key, value, key_pages, value_pages, slot_mapping, scales[0], scales[1],
            key.stride(0), value.stride(0), ROW=row, BLOCK=triton.next_power_of_2(row),
            FP8=fp8_scales is not None,
        )

    def paged_attention(
        self,
        query: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        block_tables: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        host: HostLists | None = None,
        num_short: tuple[int, int] = (0, 0),
        max_query_len: int | None = None,
    ) -> torch.Tensor:
        """Two launches of one kernel: the leading short sequences with a few query tokens per
        program, then the long ones with 64-row tiles."""
        if scale is None:
            scale = query.shape[-1] ** -0.5
        _, page, num_kv_heads, dim = key_pages.shape
        group = query.shape[1] // num_kv_heads
        out = torch.empty_like(query)
        num_seqs = context_lens.shape[0]
        short_seqs, _, all_short = split_short(cu_seqlens_q, host, num_short, max_query_len)
        if all_short:
            short_len = min(max_query_len or SHORT_QUERY, SHORT_QUERY)
            long_len = 0
        else:
            starts = cu_seqlens_q.tolist() if host is None else host.offsets
            lengths = [b - a for a, b in itertools.pairwise(starts)]
            short_len = max(lengths[:short_seqs], default=0)
            long_len = max(lengths[short_seqs:], default=0)

        def launch(seq_offset: int, count: int, q_tokens: int, longest: int) -> None:
            if count == 0:
                return
            grid = (triton.cdiv(longest, q_tokens), num_kv_heads, count)
            kernels.paged_attention_kernel[grid](
                query, key_pages, value_pages, out, block_tables, cu_seqlens_q, context_lens,
                seq_offset, scale, k_scale, v_scale,
                query.stride(0), query.stride(1), out.stride(0), out.stride(1), block_tables.stride(0),
                key_pages.stride(1), key_pages.stride(2),
                GROUP=group, D=dim, BLOCK_D=max(16, triton.next_power_of_2(dim)), PAGE=page,
                Q_TOKENS=q_tokens, BLOCK_M=max(16, triton.next_power_of_2(q_tokens * group)),
                BLOCK_N=64 if dim <= 128 else 32,
                FP8=key_pages.dtype == torch.uint8, IEEE=query.dtype == torch.float32,
            )

        if all_short:
            launch(0, num_seqs, triton.next_power_of_2(short_len), short_len)
            return out
        launch(0, short_seqs, triton.next_power_of_2(max(short_len, 1)), short_len)
        launch(short_seqs, num_seqs - short_seqs, max(1, 64 // group), long_len)
        return out
