"""A request in flight: its prompt, output and proposed tokens, how far the KV cache holds them,
and its pages."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mini_vllm.sampling_parameters import SamplingParams

if TYPE_CHECKING:
    from mini_vllm.engine.block_manager import BlockTable

__all__ = ["Request", "RequestStatus"]


class RequestStatus(enum.Enum):
    WAITING = "waiting"    # queued, or preempted and queued again
    RUNNING = "running"
    FINISHED = "finished"


_next_request_id = itertools.count()


@dataclass(eq=False)
class Request:
    """A request in flight: its tokens, how many of them the KV cache holds, its pages.

    token_ids is prompt + output + unverified proposals, and num_computed_tokens says how
    much of it is cached; the difference is what the next iteration can compute.
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    max_tokens: int = 16
    stop_token_ids: frozenset[int] = frozenset()
    request_id: int = field(default_factory=lambda: next(_next_request_id))

    output_token_ids: list[int] = field(default_factory=list)
    # A draft's guesses, awaiting verification: attended over, but not yet output.
    proposed_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0  # prompt tokens the prefix cache served when it was admitted
    status: RequestStatus = RequestStatus.WAITING
    block_table: BlockTable | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids + self.proposed_token_ids

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.proposed_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self) - self.num_computed_tokens

    def is_prefill(self) -> bool:
        """True while any prompt token is still uncomputed, chunked or not."""
        return self.num_computed_tokens < len(self.prompt_token_ids)

    def is_done(self) -> bool:
        if len(self.output_token_ids) >= self.max_tokens:
            return True
        return bool(self.output_token_ids) and self.output_token_ids[-1] in self.stop_token_ids

    def accept(self, tokens: list[int], num_accepted: int) -> int:
        """Commit a verified step: the accepted proposals, then the token the target drew
        after them, stopping early if one of them finishes the request. Returns how many
        of this step's slots to give back.

        The pass computed the pending token and every proposal; what stays cached is the
        pending token and the proposals that were both accepted and kept.
        """
        num_proposed = len(self.proposed_token_ids)
        self.proposed_token_ids = []
        kept = 0
        for index, token in enumerate(tokens):
            self.output_token_ids.append(token)
            kept += index < num_accepted
            if self.is_done():
                break
        self.num_computed_tokens += 1 + kept
        return num_proposed - kept
