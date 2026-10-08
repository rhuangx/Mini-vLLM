"""FP8 (e4m3) KV quantization with static per-tensor scales, stored as uint8."""

from __future__ import annotations

import torch

__all__ = ["FP8_MAX", "dequantize_fp8", "quantize_fp8"]

# The largest finite e4m3 value. Torch's cast turns anything past it into NaN, so quantize
# saturates first, which is what mx.to_fp8 used to do.
FP8_MAX = 448.0


def quantize_fp8(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """x / scale as e4m3 bits in uint8: half the bytes of bf16, 3 mantissa bits."""
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_fp8(x: torch.Tensor, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The inverse of quantize_fp8: decode the uint8 bits, multiply the scale back in."""
    return (x.view(torch.float8_e4m3fn).float() * scale).to(dtype)
