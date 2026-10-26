"""kv_cache.py: the dense cache appends like concatenation."""

import torch

from mini_vllm.reference.kv_cache import KvFullCache


def test_the_dense_cache_appends_like_concat():
    cache = KvFullCache()
    first = torch.randn(1, 2, 3, 8)
    second = torch.randn(1, 2, 1, 8)

    keys, values = cache.update_and_fetch(first, -first)
    assert cache.offset == 3 and keys.shape[-2] == 3

    keys, values = cache.update_and_fetch(second, -second)
    assert cache.offset == 4
    assert torch.equal(keys, torch.cat([first, second], dim=-2))
    assert torch.equal(values, -keys)
