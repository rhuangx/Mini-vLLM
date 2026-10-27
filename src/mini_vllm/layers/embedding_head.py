"""The token embedding and the LM head, both split over tensor-parallel ranks by vocabulary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_vllm.utils.distributed import all_reduce, gather_to_first, tp_rank, tp_size

__all__ = ["ParallelLMHead", "VocabParallelEmbedding"]


class VocabParallelEmbedding(nn.Module):
    """Each rank holds a contiguous slice of the vocabulary's rows; a token another rank owns
    looks up zeros here, and the all-reduce fills it in."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        assert num_embeddings % tp_size() == 0, f"vocabulary {num_embeddings} over {tp_size()} ranks"
        self.num_embeddings = num_embeddings
        self.shard = num_embeddings // tp_size()
        self.start = tp_rank() * self.shard
        self.weight = nn.Parameter(torch.empty(self.shard, embedding_dim), requires_grad=False)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.narrow(0, self.start, self.shard))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if tp_size() == 1:
            return F.embedding(ids, self.weight)
        local = ids - self.start
        mine = (local >= 0) & (local < self.shard)
        out = F.embedding(torch.where(mine, local, 0), self.weight) * mine.unsqueeze(-1)
        return all_reduce(out)


class ParallelLMHead(VocabParallelEmbedding):
    """Hidden states to logits: each rank scores its vocabulary slice, and rank 0, which
    samples, gathers them; other ranks get None."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | None:
        return gather_to_first(F.linear(hidden, self.weight))
