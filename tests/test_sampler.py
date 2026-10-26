"""sampler.py: greedy, temperature, top-k and top-p truncation, per-row params, the draw."""

import math

import pytest
import torch
from utils import assert_allclose

from mini_vllm.layers.sampler import sample, sampling_probabilities
from mini_vllm.sampling_parameters import SamplingParams

# Probabilities 0.5, 0.3, 0.15, 0.05, stored out of rank order to exercise the sort.
PROBABILITIES = [0.15, 0.5, 0.05, 0.3]
LOGITS = torch.tensor([[math.log(p) for p in PROBABILITIES]])


def test_greedy_rows_are_one_hot_and_ignore_the_generator():
    logits = torch.randn(4, 100)
    greedy = SamplingParams(temperature=0.0)

    probabilities = sampling_probabilities(logits, greedy)
    assert torch.equal(torch.argmax(probabilities, dim=-1), torch.argmax(logits, dim=-1))
    assert torch.equal(torch.sum(probabilities, dim=-1), torch.ones(4))

    for seed in range(3):
        tokens = sample(logits, greedy, generator=torch.Generator().manual_seed(seed))
        assert tokens.dtype == torch.int32
        assert torch.equal(tokens, torch.argmax(logits, dim=-1))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_untruncated_rows_are_softmax_over_temperature(temperature):
    logits = 3 * torch.randn(3, 50)
    probabilities = sampling_probabilities(logits, SamplingParams(temperature=temperature))
    assert_allclose(probabilities, torch.softmax(logits / temperature, dim=-1))


@pytest.mark.parametrize("top_k", [1, 5, 17])
def test_top_k_keeps_exactly_k_tokens(top_k):
    logits = torch.randn(3, 50)
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=top_k))

    kept = probabilities > 0
    assert kept.sum(dim=-1).tolist() == [top_k] * 3
    # The kept tokens are the k largest logits.
    threshold = torch.sort(logits, dim=-1).values[:, -top_k][:, None]
    assert torch.equal(kept, logits >= threshold)


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (0.75, [0.0, 0.5 / 0.8, 0.0, 0.3 / 0.8]),            # 0.5 < p, 0.5 + 0.3 reaches it
        (0.85, [0.15 / 0.95, 0.5 / 0.95, 0.0, 0.3 / 0.95]),  # the third token crosses p
        (0.01, [0.0, 1.0, 0.0, 0.0]),                        # the top token is always kept
    ],
)
def test_top_p_keeps_the_smallest_prefix_reaching_p(top_p, expected):
    probabilities = sampling_probabilities(LOGITS, SamplingParams(top_p=top_p))
    assert_allclose(probabilities, torch.tensor([expected]))


def test_per_row_parameters_apply_per_row():
    logits = torch.randn(3, 40)
    params = [SamplingParams(temperature=0.0), SamplingParams(top_k=3), SamplingParams(top_p=0.5)]

    together = sampling_probabilities(logits, params)
    for row, row_params in enumerate(params):
        assert_allclose(together[row : row + 1], sampling_probabilities(logits[row : row + 1], row_params))


def test_a_seeded_generator_reproduces_the_draw():
    logits = torch.randn(64, 100)
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    assert torch.equal(sample(logits, generator=first), sample(logits, generator=second))


def test_draws_follow_the_truncated_distribution():
    draws = 20_000
    params = SamplingParams(top_k=3)
    tokens = sample(LOGITS.expand(draws, -1), params, generator=torch.Generator().manual_seed(0))

    frequencies = [(tokens == token).sum().item() / draws for token in range(4)]
    expected = sampling_probabilities(LOGITS, params)[0].tolist()
    # Three standard errors of a 20k-draw frequency is under 0.011 for any p.
    assert all(abs(f - p) < 0.011 for f, p in zip(frequencies, expected, strict=True)), frequencies
