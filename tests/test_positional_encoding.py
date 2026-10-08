"""positional_encoding.py against transformers.apply_rotary_pos_emb."""

import math

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from utils import DTYPES, assert_allclose, requires_cuda

from mini_vllm.positional_encoding import RoPE

THETA = 1_000_000.0


def fast_rope(x: torch.Tensor, offset: int) -> torch.Tensor:
    """apply_rotary_pos_emb on a B x L x H x D input, with the same half-split tables."""
    length, head_dim = x.shape[1], x.shape[-1]
    positions = torch.arange(offset, offset + length, device=x.device)
    inverse = 1.0 / (THETA ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    freqs = positions.float()[:, None] * inverse
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]  # 1 x L x D, unsqueezed over heads below
    rotated, _ = apply_rotary_pos_emb(x.float(), x.float()[:, :, :1], cos, sin, unsqueeze_dim=2)
    return rotated.to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 17])
def test_rope_matches_transformers(dtype, offset):
    rope = RoPE(head_dim=128, max_seq_len=64, theta=THETA)
    x = torch.randn(2, 9, 4, 128).to(dtype)

    assert_allclose(rope(x, torch.arange(offset, offset + 9)), fast_rope(x, offset))


@pytest.mark.parametrize("position", [4000, 30000])
def test_long_positions_stay_within_the_fp32_angle_error(position):
    # An fp32 angle pos * theta^(-2i/D) is off by ~pos * 2^-24, so neither this nor
    # apply_rotary_pos_emb can meet the op tolerance far out; both are held to an exact reference.
    head_dim = 128
    x = torch.randn(1, 1, 1, head_dim)
    values, half = x.reshape(-1).tolist(), head_dim // 2

    exact = [0.0] * head_dim
    for i in range(half):
        angle = position * THETA ** (-2 * i / head_dim)
        cos, sin = math.cos(angle), math.sin(angle)
        exact[i] = values[i] * cos - values[i + half] * sin
        exact[i + half] = values[i + half] * cos + values[i] * sin

    bound = position * 2**-20 * max(abs(v) for v in values)
    for rotated in (RoPE(head_dim, position + 1, THETA)(x, torch.tensor([position])), fast_rope(x, position)):
        worst = max(abs(a - b) for a, b in zip(rotated.reshape(-1).tolist(), exact, strict=True))
        assert worst < bound, f"{worst:.3g} >= {bound:.3g}"


def test_per_row_positions_rotate_each_row_independently():
    rope = RoPE(head_dim=64, max_seq_len=256, theta=THETA)
    x = torch.randn(3, 5, 2, 64)
    offsets = [0, 40, 200]
    positions = torch.tensor([[offset + i for i in range(5)] for offset in offsets])

    out = rope(x, positions)
    for row in range(len(offsets)):
        assert torch.equal(out[row : row + 1], rope(x[row : row + 1], positions[row]))


def test_positions_are_explicit_not_assumed():
    # A gap in the positions must rotate each token by its own position, not its index.
    rope = RoPE(head_dim=64, max_seq_len=256, theta=THETA)
    x = torch.randn(1, 4, 2, 64)
    positions = [0, 1, 2, 7]

    out = rope(x, torch.tensor(positions))
    for i, position in enumerate(positions):
        assert_allclose(out[:, i : i + 1], fast_rope(x[:, i : i + 1], position))


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("per_row", [False, True])
def test_the_rope_kernel_matches_the_oracle(dtype, per_row):
    """The kernel gathers the same fp32 tables, so even far positions agree to rounding."""
    rope = RoPE(head_dim=128, max_seq_len=40_000, theta=THETA, device=torch.device("cuda"))
    cuda = RoPE(head_dim=128, max_seq_len=40_000, theta=THETA, use_cuda=True, device=torch.device("cuda"))
    x = torch.randn(2, 7, 4, 128, device="cuda").to(dtype)
    far = torch.tensor([[39_000 + 3 * i + b for i in range(7)] for b in range(2)], device="cuda")
    positions = far if per_row else torch.arange(5, 12, device="cuda")
    assert_allclose(cuda(x, positions), rope(x, positions))
