"""Where an operator runs: pure PyTorch, or the CUDA kernels in src/extensions/.

Every operator the kernels cover goes through a Backend. The pure-PyTorch math stays in the
layer modules, as the oracle every kernel is tested against; TorchBackend calls it, and
CudaBackend overrides each operator with its kernel, falling back to the pure path for the
shapes a kernel does not take. Backends are imported only when first asked for, so the layers
can import this module without a cycle.
"""

from __future__ import annotations

from typing import Protocol

import torch

__all__ = ["NAMES", "Backend", "get_backend"]

NAMES = ("torch", "cuda")


class Backend(Protocol):
    name: str

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """silu(gate) * up."""
        ...

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """x * rsqrt(mean(x^2) + eps) * weight over the last axis."""
        ...

    def rope(
        self, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """x [B, L, H, D] rotated by the fp32 table rows at positions [L] or [B, L]."""
        ...

    def dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """Grouped attention: query B x H_q x L x D over key, value B x H_k x S x D."""
        ...

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
        host: tuple[list[list[int]], list[int], list[int]] | None = None,
    ) -> torch.Tensor:
        """Causal grouped attention for a ragged batch, T x H_q x D, over its pages."""
        ...

    def write_kv(
        self,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fp8_scales: tuple[float, float] | None = None,
    ) -> None:
        """Scatter key and value [T, H_k, D] into their pages' slots, in place; quantized to
        fp8 with (k_scale, v_scale) when fp8_scales is given, raw otherwise."""
        ...


_backends: dict[str, Backend] = {}


def get_backend(backend: str | Backend = "torch") -> Backend:
    """The backend called `backend`, created once; a Backend passes through. Asking for "cuda"
    before src/extensions is built fails here, not halfway through a forward."""
    if not isinstance(backend, str):
        return backend
    if backend not in _backends:
        if backend == "torch":
            from mini_vllm.layers.backends.torch_backend import TorchBackend

            _backends[backend] = TorchBackend()
        elif backend == "cuda":
            from mini_vllm.layers.backends.cuda_backend import CudaBackend

            _backends[backend] = CudaBackend()
        else:
            raise ValueError(f"unknown backend {backend!r}; expected one of {NAMES}")
    return _backends[backend]
