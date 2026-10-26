"""utils/context.py: forwards nest and restore, and a self-draft caches in its own pool.

These build their tiny model on whatever device there is, so they run on a Mac too.
"""

import pytest
import torch
from conftest import TINY_QWEN3
from transformers import Qwen3Config, Qwen3ForCausalLM

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.forward_batch import ForwardBatch
from mini_vllm.engine.kv_cache import PagedKvPool
from mini_vllm.engine.model_runner import run_batch
from mini_vllm.engine.sequence import Request
from mini_vllm.engine.speculative import self_draft
from mini_vllm.reference.generate import forward_logits
from mini_vllm.reference.kv_cache import KvFullCache
from mini_vllm.utils.context import Context, forward_context, get_context, reset_context
from mini_vllm.utils.loader import from_hf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    hf = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3)).to(device=DEVICE, dtype=torch.float32).eval()
    return from_hf(hf)


def pool_for(model, num_layers=None) -> PagedKvPool:
    c = model.config
    return PagedKvPool(num_layers or c.num_hidden_layers, 16, 4, c.num_key_value_heads, c.head_dim,
                       dtype=torch.float32, device=DEVICE)


def scheduled_batch(kv: PagedKvPool, prompt: list[int]) -> ForwardBatch:
    manager = BlockManager(kv)
    request = Request(prompt_token_ids=prompt)
    manager.allocate(request, len(prompt))
    return ForwardBatch.from_scheduled([(request, len(prompt))], manager)


def test_contexts_nest_and_restore_even_on_an_error():
    outer, inner = Context(dense_caches=[]), Context(dense_caches=[KvFullCache()])
    with forward_context(outer):
        with pytest.raises(RuntimeError), forward_context(inner):
            assert get_context() is inner
            raise RuntimeError("a forward that fails")
        assert get_context() is outer
    assert get_context() == Context()


def test_reset_clears_the_context():
    with forward_context(Context(dense_caches=[])):
        reset_context()
        assert get_context() == Context()
    assert get_context() == Context()


def test_a_self_draft_caches_in_its_own_pool_and_leaves_the_target_pool_alone(model):
    """The draft shares the target's first layer objects: what it writes must land in the
    draft's pool, because the pool comes with the forward rather than with the layer."""
    prompt = [3, 1, 4, 1, 5, 9, 2]
    target_kv, draft_kv = pool_for(model), pool_for(model, num_layers=1)
    draft = self_draft(model, 1)
    assert draft.layers[0] is model.layers[0]

    target_logits = run_batch(model, target_kv, scheduled_batch(target_kv, prompt))
    before = [pages.clone() for pages in (*target_kv.keys, *target_kv.values)]
    draft_logits = run_batch(draft, draft_kv, scheduled_batch(draft_kv, prompt))

    assert all(torch.equal(a, b) for a, b in zip(before, (*target_kv.keys, *target_kv.values), strict=True))
    assert draft_kv.keys[0].abs().sum() > 0, "the draft wrote nothing into its own pool"
    # Its one layer wrote what the target's first layer did, into the same slots.
    torch.testing.assert_close(draft_kv.keys[0], target_kv.keys[0])
    assert not torch.allclose(draft_logits, target_logits), "a one-layer draft matched the full model"
    assert get_context() == Context()


def test_a_pool_shallower_than_the_model_is_refused(model):
    shallow = pool_for(model, num_layers=1)
    with pytest.raises(AssertionError, match="layers over a 1-layer pool"):
        run_batch(model, shallow, scheduled_batch(shallow, [1, 2, 3]))


def test_the_paged_forward_matches_the_dense_one(model):
    prompt = [7, 7, 2, 9, 4, 4, 1, 8, 3]
    kv = pool_for(model)
    paged = run_batch(model, kv, scheduled_batch(kv, prompt))
    ids = torch.tensor([prompt], device=DEVICE)
    dense = forward_logits(model, ids, torch.arange(len(prompt), device=DEVICE))[0]
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-5)
