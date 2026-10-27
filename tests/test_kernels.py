"""Every GPU backend's operators against the pure-PyTorch ones, and the pure paged attention
against plain causal attention over a contiguous history."""

import pytest
import torch
from conftest import GPU_BACKENDS

from mini_vllm.engine.kv_cache import PagedKvPool
from mini_vllm.layers.attention import scaled_dot_product_attention_grouped
from mini_vllm.layers.backends import get_backend

TORCH = get_backend("torch")
gpu = pytest.mark.parametrize("backend", GPU_BACKENDS)
DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])


def close(actual, expected):
    tolerance = 1e-5 if actual.dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tolerance, atol=tolerance)


def ragged(lengths, histories, heads=4, kv_heads=2, dim=32, page=4, dtype=torch.float32, device="cpu",
           fp8=False):
    """A pool holding each sequence's history at shuffled pages, and its batch metadata."""
    pool = PagedKvPool(1, 64, page, kv_heads, dim, dtype=dtype, device=device, fp8=fp8, k_scale=0.5,
                       v_scale=0.25)
    order = torch.randperm(64).tolist()
    tables, starts, contexts = [], [0], []
    for length, history in zip(lengths, histories, strict=True):
        context = length + history
        blocks = [order.pop() for _ in range(-(-context // page))]
        slots = torch.tensor([blocks[i // page] * page + i % page for i in range(context)], device=device)
        TORCH.write_kv(pool.keys[0], pool.values[0], slots.int(), torch.randn(context, kv_heads, dim, device=device),
                       torch.randn(context, kv_heads, dim, device=device), (0.5, 0.25) if fp8 else None)
        tables.append(blocks + [-1] * (16 - len(blocks)))
        starts.append(starts[-1] + length)
        contexts.append(context)
    ints = {"dtype": torch.int32, "device": device}
    query = torch.randn(starts[-1], heads, dim, dtype=dtype, device=device)
    key_pages, value_pages = pool.pages(0)
    return (query, key_pages, value_pages, torch.tensor(tables, **ints), torch.tensor(starts, **ints),
            torch.tensor(contexts, **ints))


def test_pure_paged_attention_is_causal_attention_over_the_history():
    query, key_pages, value_pages, tables, starts, contexts = ragged([5], [7])
    keys = key_pages.reshape(-1, 2, 32)[(tables[0, :3].long()[:, None] * 4 + torch.arange(4)).reshape(-1)][:12]
    values = value_pages.reshape(-1, 2, 32)[(tables[0, :3].long()[:, None] * 4 + torch.arange(4)).reshape(-1)][:12]
    dense = scaled_dot_product_attention_grouped(
        query.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), mask="causal"
    ).transpose(0, 1)
    close(TORCH.paged_attention(query, key_pages, value_pages, tables, starts, contexts), dense)


@gpu
@DTYPES
def test_elementwise_operators_match_torch(backend, dtype):
    b = get_backend(backend)
    x, residual = torch.randn(7, 64, dtype=dtype, device="cuda"), torch.randn(7, 64, dtype=dtype, device="cuda")
    weight = torch.randn(64, dtype=dtype, device="cuda")
    close(b.rms_norm(x, weight, 1e-6), TORCH.rms_norm(x, weight, 1e-6))
    for got, want in zip(b.add_rms_norm(x, residual, weight, 1e-6), TORCH.add_rms_norm(x, residual, weight, 1e-6),
                         strict=True):
        close(got, want)
    gate_up = torch.randn(7, 2 * 96, dtype=dtype, device="cuda")
    close(b.silu_and_mul(gate_up), TORCH.silu_and_mul(gate_up))
    q = torch.randn(1, 7, 4, 32, dtype=dtype, device="cuda")
    angles = torch.outer(torch.arange(300.0), torch.rand(16)).repeat(1, 2).cuda()
    positions = torch.tensor([0, 1, 2, 3, 150, 151, 299], device="cuda")
    close(b.rope(q, positions, angles.cos(), angles.sin()), TORCH.rope(q, positions, angles.cos(), angles.sin()))


@gpu
@pytest.mark.parametrize("fp8", [False, True])
def test_kv_writes_match_torch_and_skip_padding(backend, fp8):
    pools = [PagedKvPool(1, 8, 4, 2, 32, dtype=torch.bfloat16, device="cuda", fp8=fp8, k_scale=0.5, v_scale=0.25)
             for _ in range(2)]
    key = torch.randn(4, 2, 32, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(4, 2, 32, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([5, 17, 3, 30], dtype=torch.int32, device="cuda")
    scales = (0.5, 0.25) if fp8 else None
    TORCH.write_kv(pools[0].keys[0], pools[0].values[0], slots, key, value, scales)
    get_backend(backend).write_kv(pools[1].keys[0], pools[1].values[0], slots, key, value, scales)
    assert torch.equal(pools[0].keys[0], pools[1].keys[0]) and torch.equal(pools[0].values[0], pools[1].values[0])
    before = pools[1].keys[0].clone()
    get_backend(backend).write_kv(pools[1].keys[0], pools[1].values[0], torch.full_like(slots, -1), key, value, scales)
    assert torch.equal(before, pools[1].keys[0]), "a padded row (slot -1) wrote to the pool"


@gpu
@DTYPES
@pytest.mark.parametrize("fp8", [False, True])
def test_paged_attention_matches_torch(backend, dtype, fp8):
    """Decodes, a speculative verify, a chunk resuming over history and a fresh prefill, in one
    batch, short ones first; and a decode-only batch."""
    for lengths, histories in (([1, 1, 4, 13, 20], [9, 30, 6, 11, 0]), ([1, 1, 1], [3, 40, 17])):
        args = ragged(lengths, histories, dtype=dtype, device="cuda", fp8=fp8)
        scales = {"k_scale": 0.5, "v_scale": 0.25}
        close(get_backend(backend).paged_attention(*args, **scales), TORCH.paged_attention(*args, **scales))
