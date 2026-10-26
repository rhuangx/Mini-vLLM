"""The tokenizer the engine uses, wrapped from Hugging Face, and the ids that stop generation."""

from __future__ import annotations

from typing import Any

__all__ = ["Tokenizer", "eos_token_ids"]


class Tokenizer:
    """The slice of a Hugging Face tokenizer the engine and the benches use.

    eos_token_ids is a plain list so a bench can disable stopping by assigning []. encode,
    decode and vocab forward to the wrapped tokenizer, which is .hf; len(tokenizer.vocab) is
    the vocabulary.
    """

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

    @property
    def vocab(self) -> dict:
        return self.hf.get_vocab()


def eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """generation_config.eos_token_id, which may be one id or several."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)
