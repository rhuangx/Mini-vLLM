"""Where an operator runs: pure PyTorch, the CUDA kernels in src/extensions/, or Triton.

The pure-PyTorch math stays in the layer modules, as the oracle every kernel is tested
against; TorchBackend calls it, and the GPU backends override each operator with a kernel,
falling back to the pure path for shapes a kernel does not take. Backends are imported only
when first asked for, so the layers can import this module without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from mini_vllm.engine.model_runner import HostLists

__all__ = ["NAMES", "SHORT_QUERY", "Backend", "get_backend"]

NAMES = ("torch", "cuda", "triton")

# Decode kernels give every query token its own work, so they take sequences of at most this
# many query tokens: decodes and speculative verifies.
SHORT_QUERY = 8


class Backend(Protocol):
    name: str
    supports_graphs: bool  # every operator capturable in a CUDA graph: no host syncs

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor: ...

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...

    def add_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(rms_norm(x + residual), x + residual)."""
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
        host: HostLists | None = None,
        num_short: tuple[int, int] = (0, 0),
        max_query_len: int | None = None,
    ) -> torch.Tensor:
        """Causal grouped attention for a ragged batch, T x H_q x D, over its pages. The first
        num_short = (sequences, tokens) are short (decodes), the rest long; max_query_len, when
        known on the host, spares a sync."""
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
        fp8 with (k_scale, v_scale) when fp8_scales is given. A slot of -1 is skipped."""
        ...


_backends: dict[str, Backend] = {}


def get_backend(backend: str | Backend = "torch") -> Backend:
    """The backend called `backend`, created once; a Backend passes through. Asking for a GPU
    backend that cannot run here fails at once, not halfway through a forward."""
    if not isinstance(backend, str):
        return backend
    if backend not in _backends:
        if backend == "torch":
            from mini_vllm.layers.backends.torch_backend import TorchBackend as Chosen
        elif backend == "cuda":
            from mini_vllm.layers.backends.cuda_backend import CudaBackend as Chosen
        elif backend == "triton":
            from mini_vllm.layers.backends.triton_backend import TritonBackend as Chosen
        else:
            raise ValueError(f"unknown backend {backend!r}; expected one of {NAMES}")
        _backends[backend] = Chosen()
    return _backends[backend]
