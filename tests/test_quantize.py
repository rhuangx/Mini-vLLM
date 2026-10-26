"""quantize.py: the e4m3 error bound, saturation, and the scale."""

import pytest
import torch

from mini_vllm.layers.quantization import FP8_MAX, dequantize_fp8, quantize_fp8


@pytest.mark.parametrize("scale", [1.0, 0.05])
def test_round_trip_error_is_within_the_e4m3_bound(scale):
    x = (scale * 50 * torch.randn(4096)).float()
    back = dequantize_fp8(quantize_fp8(x, scale), scale, torch.float32)

    # 3 mantissa bits round to within 2^-4 relative; subnormals to half their 2^-9 step.
    bound = 2**-4 * torch.abs(x) + scale * 2**-10
    assert torch.all(torch.abs(back - x) <= bound)


def test_out_of_range_values_saturate():
    x = torch.tensor([1e6, -1e6, 500.0, FP8_MAX])
    back = dequantize_fp8(quantize_fp8(x), 1.0, torch.float32)
    assert back.tolist() == [FP8_MAX, -FP8_MAX, FP8_MAX, FP8_MAX]


def test_the_scale_divides_before_the_cast():
    x = torch.randn(256).bfloat16()
    bits = quantize_fp8(x, 0.25)

    assert bits.dtype == torch.uint8 and bits.nbytes == x.numel()
    assert torch.equal(bits, quantize_fp8(x.float() / 0.25))
