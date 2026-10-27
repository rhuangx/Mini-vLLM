"""Loading: build the model from its checkpoint's config, then stream its tensors, from
safetensors or a transformers model's state dict, through each parameter's weight_loader.
transformers supplies only the config, the tokenizer and the generation config."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mini_vllm.config import ModelConfig
from mini_vllm.layers.backends import Backend
from mini_vllm.models import MODELS

__all__ = [
    "DEFAULT_MODEL",
    "Tokenizer",
    "build_model",
    "eos_token_ids",
    "from_hf",
    "iter_safetensors",
    "iter_state_dict",
    "load",
    "load_weights",
]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class Tokenizer:
    """The slice of a Hugging Face tokenizer the engine uses; the tokenizer itself is .hf.
    eos_token_ids is a plain list, so stopping can be turned off by assigning []."""

    def __init__(self, tokenizer: Any, eos_token_ids: list[int] | None = None) -> None:
        self.hf = tokenizer
        if eos_token_ids is None:
            eos = tokenizer.eos_token_id
            eos_token_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        self.eos_token_ids = eos_token_ids

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self.hf.encode(text, **kwargs)

    def decode(self, ids, **kwargs: Any) -> str:
        return self.hf.decode(ids, **kwargs)


def eos_token_ids(source: Any, tokenizer: Any) -> list[int]:
    """The generation config's eos_token_id (of a model, or the config itself), which may be
    one id or several; the tokenizer's when it has none."""
    config = getattr(source, "generation_config", source)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_model(config: ModelConfig, dtype: torch.dtype, device: torch.device | str,
                backend: str | Backend = "torch") -> nn.Module:
    """The architecture the config names, with uninitialized weights in dtype on device."""
    with _default_dtype(dtype), torch.device(device):
        return MODELS[config.architecture](config, backend)


def _default_weight_loader(param: nn.Parameter, loaded: torch.Tensor) -> None:
    param.data.copy_(loaded)


def load_weights(model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Copy every checkpoint tensor into its parameter through the parameter's weight_loader,
    packed ones (q/k/v, gate/up) with the shard they fill; then insist every parameter, and
    every shard of a packed one, was filled."""
    params = dict(model.named_parameters())
    packed = getattr(model, "packed_modules_mapping", {})
    filled: set[tuple[str, object]] = set()
    for name, tensor in weights:
        if name.endswith(".inv_freq") or (name == "lm_head.weight" and model.config.tie_word_embeddings):
            continue  # recomputed tables, and the tied head, which is the embedding
        for source, (target, shard) in packed.items():
            if f".{source}." in name:
                key = name.replace(f".{source}.", f".{target}.")
                params[key].weight_loader(params[key], tensor, shard)
                filled.add((key, shard))
                break
        else:
            if name not in params:
                raise KeyError(f"checkpoint tensor {name} matches no parameter")
            getattr(params[name], "weight_loader", _default_weight_loader)(params[name], tensor)
            filled.add((name, None))
    wanted = set()
    for name in params:
        shards = {shard for _, (target, shard) in packed.items() if f".{target}." in name}
        wanted |= {(name, shard) for shard in shards} if shards else {(name, None)}
    if missing := wanted - filled:
        raise KeyError(f"no checkpoint tensor for {sorted(missing)[:5]}")


def iter_state_dict(hf_model: Any) -> Iterator[tuple[str, torch.Tensor]]:
    yield from hf_model.state_dict().items()


def iter_safetensors(path: str | Path, device: torch.device | str) -> Iterator[tuple[str, torch.Tensor]]:
    """Every tensor in the checkpoint's safetensors files, read straight onto device."""
    from safetensors import safe_open

    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                yield name, f.get_tensor(name)


def from_hf(hf_model: Any, backend: str | Backend = "torch") -> nn.Module:
    """Our model with a transformers model's weights, in its dtype on its device: the parity
    tests' way in, through the same loader as a checkpoint."""
    weight = next(hf_model.parameters())
    model = build_model(ModelConfig.from_hf_config(hf_model.config), weight.dtype, weight.device, backend)
    load_weights(model, iter_state_dict(hf_model))
    return model


def load(model: str = DEFAULT_MODEL, backend: str | Backend = "torch",
         device: torch.device | str | None = None) -> tuple[nn.Module, Tokenizer]:
    """A bf16 checkpoint, a local directory or a hub name, as (model, tokenizer); no
    transformers model is ever built."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    path = model if Path(model).is_dir() else snapshot_download(
        model, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"]
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig.from_hf_config(AutoConfig.from_pretrained(path))
    built = build_model(config, torch.bfloat16, device, backend)
    load_weights(built, iter_safetensors(path, device))
    hf_tokenizer = AutoTokenizer.from_pretrained(path)
    try:
        generation = GenerationConfig.from_pretrained(path)
    except OSError:
        generation = None
    return built, Tokenizer(hf_tokenizer, eos_token_ids(generation, hf_tokenizer))
