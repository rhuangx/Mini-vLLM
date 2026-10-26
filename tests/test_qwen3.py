"""qwen3.py and models.py: the transplanted model against transformers, and the cache ladder."""

import pytest
import torch
from conftest import make_tiny_qwen3
from utils import assert_allclose

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.forward_batch import ForwardBatch
from mini_vllm.engine.kv_cache import PagedKvPool
from mini_vllm.engine.sequence import Request
from mini_vllm.reference.kv_cache import KvFullCache
from mini_vllm.utils.loader import from_hf


def caches_for(model):
    return [KvFullCache() for _ in model.layers]


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_logits_match_transformers(tie_word_embeddings):
    reference = make_tiny_qwen3(tie_word_embeddings=tie_word_embeddings)
    model = from_hf(reference)
    ids = torch.randint(0, 512, (2, 10), device="cuda")

    with torch.no_grad():
        assert_allclose(model(ids, torch.arange(10, device="cuda")), reference(ids).logits)


def test_cached_prefill_matches_the_dense_forward(tiny_qwen3):
    model = from_hf(tiny_qwen3)
    ids = torch.randint(0, 512, (1, 10), device="cuda")

    positions = torch.arange(10, device="cuda")
    assert_allclose(model(ids, positions, caches_for(model)), model(ids, positions))


def test_cached_decode_matches_recomputing_from_scratch(tiny_qwen3):
    """The whole point of the cache: one token forwarded, same logits as a full pass."""
    model = from_hf(tiny_qwen3)
    prompt = torch.randint(0, 512, (1, 9), device="cuda")
    step = torch.randint(0, 512, (1, 1), device="cuda")

    caches = caches_for(model)
    model(prompt, torch.arange(9, device="cuda"), caches)
    got = model(step, torch.tensor([9], device="cuda"), caches)
    want = model(torch.cat([prompt, step], dim=1), torch.arange(10, device="cuda"))[:, -1:]

    assert_allclose(got, want)


@pytest.mark.parametrize("chunk", [1, 7, 49, 50])  # single tokens, uneven, all but one, all
def test_chunked_prefill_matches_one_pass(tiny_qwen3, chunk):
    """Feeding the prompt in pieces must give every position the same logits."""
    model = from_hf(tiny_qwen3)
    prompt = torch.randint(0, 512, (1, 50), device="cuda")
    whole = model(prompt, torch.arange(50, device="cuda"))

    caches = caches_for(model)
    pieces = [
        model(
            prompt[:, start : start + chunk],
            torch.arange(start, min(start + chunk, 50), device="cuda"),
            caches,
        )
        for start in range(0, 50, chunk)
    ]
    assert_allclose(torch.cat(pieces, dim=1), whole)


def test_a_wrong_position_at_a_chunk_boundary_is_caught(tiny_qwen3):
    """Restarting RoPE at zero on the second chunk must change the logits."""
    model = from_hf(tiny_qwen3)
    prompt = torch.randint(0, 512, (1, 40), device="cuda")
    whole = model(prompt, torch.arange(40, device="cuda"))[:, -1]

    caches = caches_for(model)
    model(prompt[:, :20], torch.arange(20, device="cuda"), caches)
    wrong = model(prompt[:, 20:], torch.arange(20, device="cuda"), caches)[:, -1]

    assert not torch.allclose(wrong, whole, rtol=1e-3, atol=1e-3), (
        "restarting RoPE at zero changed nothing, so the positions are not being used"
    )


def test_real_logits_sit_close_to_transformers(real_qwen3):
    reference, tokenizer = real_qwen3
    model = from_hf(reference)
    text = "Paged attention stores the KV cache in fixed-size blocks"
    ids = torch.tensor([tokenizer.encode(text)], device="cuda")
    length = ids.shape[1]

    ours = model(ids, torch.arange(length, device="cuda")).float()
    with torch.no_grad():
        theirs = reference(ids).logits.float()

    # BF16 through 28 layers: bound the relative error, not each element.
    relative = (torch.linalg.norm(ours - theirs) / torch.linalg.norm(theirs)).item()
    assert relative < 0.05, relative
    # An exact tie (two logits equal in bf16) makes argmax the lower index, and one ulp
    # in our fp32 scores flips it. Require the argmax only where HF's top two differ.
    margin = torch.topk(theirs, 2, dim=-1).values
    decided = (margin[..., 0] - margin[..., 1]) > 0
    assert decided.any()
    assert torch.equal(torch.argmax(ours, dim=-1)[decided], torch.argmax(theirs, dim=-1)[decided])


def paged_setup(model, num_blocks=32, block_size=4):
    c = model.config
    kv = PagedKvPool(c.num_hidden_layers, num_blocks, block_size, c.num_key_value_heads, c.head_dim,
                     dtype=torch.float32, device=model.embedding.weight.device)
    return kv, BlockManager(kv)


def paged_forward(model, kv, manager, scheduled):
    for request, count in scheduled:
        manager.allocate(request, count)
    batch = ForwardBatch.from_scheduled(scheduled, manager)
    logits = model(batch.input_ids[None], batch.positions, kv.caches(batch))[0]
    for request, count in scheduled:
        request.num_computed_tokens += count
    return logits, batch


def test_a_ragged_paged_batch_matches_each_dense_run(tiny_qwen3):
    """The ladder's last rung: prefill two sequences, then one pass carrying a decode for
    each beside a third's prefill, every row against the full recompute of its sequence."""
    model = from_hf(tiny_qwen3)
    kv, manager = paged_setup(model)
    first, second = Request(prompt_token_ids=list(range(1, 8))), Request(prompt_token_ids=list(range(20, 25)))

    prefill, _ = paged_forward(model, kv, manager, [(first, 7), (second, 5)])
    device = model.embedding.weight.device
    alone = model(torch.tensor([first.token_ids], device=device), torch.arange(7, device=device))
    assert_allclose(prefill[:7], alone[0])

    first.output_token_ids.append(11)
    second.output_token_ids.append(12)
    third = Request(prompt_token_ids=list(range(40, 46)))
    logits, batch = paged_forward(model, kv, manager, [(first, 1), (second, 1), (third, 6)])

    starts = batch.cu_seqlens_q.tolist()
    for index, request in enumerate((first, second, third)):
        rows = logits[starts[index] : starts[index + 1]]
        ids = torch.tensor([request.token_ids], device=device)
        dense = model(ids, torch.arange(len(request), device=device))[0, len(request) - rows.shape[0] :]
        assert_allclose(rows, dense)
