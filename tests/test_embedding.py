"""embedding.py against torch.nn.Embedding."""

import pytest
import torch
from utils import DTYPES, assert_allclose

from mini_vllm.layers.embedding_head import Embedding


@pytest.mark.parametrize("dtype", DTYPES)
def test_lookup_and_tied_head_match_nn_embedding(dtype):
    reference = torch.nn.Embedding(512, 64).to(dtype)
    embedding = Embedding(512, 64, reference.weight)

    ids = torch.randint(0, 512, (3, 7))
    assert torch.equal(embedding(ids), reference(ids))

    h = torch.randn(3, 7, 64, dtype=dtype)
    assert_allclose(embedding.as_linear(h), torch.nn.functional.linear(h, reference.weight))
