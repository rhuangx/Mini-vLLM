"""The model against transformers, and the engine against itself one request at a time:
batching, chunking, preemption, prefix caching, fp8, speculation, parallel sampling, CUDA
graphs and tensor parallelism must change no token."""

import pytest
import torch
from conftest import BACKENDS, DEVICE, GPU_BACKENDS, tiny_hf

from mini_vllm import DEFAULT_MODEL, LLM, EngineConfig, SamplingParams, from_hf
from mini_vllm.utils.loader import build_model, iter_state_dict, load_weights

PROMPTS = [[1, 2, 3], [5], [7] * 9, [2, 4], [9, 8, 7, 6, 5], [1], list(range(30, 70))]


class LetterTokenizer:
    eos_token_ids = ()

    def encode(self, text):
        return [ord(c) % 512 for c in text]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def generate(hf, **config):
    config.setdefault("num_blocks", 64)
    config.setdefault("block_size", 4)
    llm = LLM(from_hf(hf), LetterTokenizer(), EngineConfig(**config))
    out = [c.token_ids for c in llm.generate(PROMPTS, max_tokens=10)]
    assert not llm.scheduler.has_work and llm.manager.pool.num_free == llm.kv.num_blocks, "pages leaked"
    return out, llm


@pytest.mark.parametrize("kind, tied", [("qwen3", False), ("qwen3", True), ("llama", False)])
def test_logits_match_transformers(kind, tied):
    hf = tiny_hf(kind, tie_word_embeddings=tied)
    model = from_hf(hf)
    ids = torch.randint(0, 512, (12,), device=DEVICE)
    with torch.no_grad():
        theirs = hf(ids[None]).logits[0]
        ours = model.compute_logits(model(ids, torch.arange(12, device=DEVICE)))
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_a_missing_checkpoint_tensor_is_refused(tiny_qwen3):
    model = build_model(from_hf(tiny_qwen3).config, torch.float32, DEVICE)
    weights = [(n, t) for n, t in iter_state_dict(tiny_qwen3) if "layers.1.mlp.up_proj" not in n]
    with pytest.raises(KeyError, match="gate_up_proj"):
        load_weights(model, weights)


CONFIGS = [
    {},
    {"max_batched_tokens": 12, "max_sequences": 3, "chunk_size": 8},  # chunked, decodes ride along
    {"num_blocks": 14, "max_batched_tokens": 16},                     # steady preemption
    {"enable_prefix_caching": True},
    {"num_speculative_tokens": 3, "num_draft_layers": 1},             # a draft mostly rejected
    {"num_speculative_tokens": 4},                                    # a full-depth self-draft
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("config", CONFIGS)
def test_every_request_matches_running_it_alone(tiny_qwen3, backend, config):
    alone, _ = generate(tiny_qwen3, max_sequences=1, enable_chunked_prefill=False)
    got, _ = generate(tiny_qwen3, backend=backend, **config)
    assert got == alone


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp8_pages_batch_like_fp8_alone(tiny_qwen3, backend):
    alone, _ = generate(tiny_qwen3, fp8_kv_cache=True, max_sequences=1, enable_chunked_prefill=False)
    assert generate(tiny_qwen3, backend=backend, fp8_kv_cache=True)[0] == alone


def test_greedy_branches_of_one_prompt_agree(tiny_qwen3):
    llm = LLM(from_hf(tiny_qwen3), LetterTokenizer(), EngineConfig(num_blocks=64, block_size=4))
    out = llm.generate([[1, 2, 3]], SamplingParams(temperature=0.0, n=3), max_tokens=6)
    assert len(out) == 3 and all(c.token_ids == out[0].token_ids for c in out)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_cuda_graphs_change_no_token(tiny_qwen3, backend):
    eager, _ = generate(tiny_qwen3, backend=backend, enforce_eager=True)
    graphed, llm = generate(tiny_qwen3, backend=backend, max_model_len=256)
    assert llm.runner.stats.graph_steps > 0
    assert graphed == eager


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_tensor_parallel_matches_one_gpu():
    prompts = [[9707, 11, 1879, 0], [785, 6722, 315, 9625, 374]]
    one = LLM.from_pretrained(DEFAULT_MODEL, num_blocks=256, enforce_eager=True)
    want = [c.token_ids for c in one.generate(prompts, max_tokens=8, ignore_eos=True)]
    del one
    two = LLM.from_pretrained(DEFAULT_MODEL, num_blocks=256, tensor_parallel_size=2)
    try:
        assert [c.token_ids for c in two.generate(prompts, max_tokens=8, ignore_eos=True)] == want
    finally:
        two.close()
