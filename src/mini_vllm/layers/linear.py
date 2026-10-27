"""Linear layers, split over tensor-parallel ranks. Each parameter carries a weight_loader that
copies its rank's shard out of a full checkpoint tensor; packed layers (q/k/v, gate/up) take a
shard id saying which slice of the packed weight a checkpoint tensor fills."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_vllm.utils.distributed import all_reduce, tp_rank, tp_size

__all__ = [
    "ColumnParallelLinear",
    "LinearBase",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
]


class LinearBase(nn.Module):
    """y = x @ W.T (+ b), W stored out x in as the checkpoint keeps it."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Each rank holds output rows [rank * out/tp, (rank + 1) * out/tp); no communication."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        assert output_size % tp_size() == 0, f"{output_size} outputs over {tp_size()} ranks"
        super().__init__(input_size, output_size // tp_size(), bias)

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        shard = param.shape[0]
        param.data.copy_(loaded.narrow(0, tp_rank() * shard, shard))


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Several column-parallel outputs side by side, e.g. gate and up; shard id i fills the
    i-th, each split over the ranks on its own."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor, shard_id: int) -> None:
        tp = tp_size()
        offset = sum(self.output_sizes[:shard_id]) // tp
        size = self.output_sizes[shard_id] // tp
        param.data.narrow(0, offset, size).copy_(loaded.narrow(0, tp_rank() * size, size))


class QKVParallelLinear(ColumnParallelLinear):
    """q, k and v projections in one GEMM, [q | k | v] per rank, each rank holding its heads."""

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_kv_heads: int,
                 bias: bool = False) -> None:
        tp = tp_size()
        assert num_heads % tp == 0 and num_kv_heads % tp == 0, f"heads {num_heads}/{num_kv_heads} over {tp}"
        self.head_dim = head_dim
        self.num_heads, self.num_kv_heads = num_heads // tp, num_kv_heads // tp
        super().__init__(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim, bias)

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor, shard_id: str) -> None:
        q, kv = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        offset, size = {"q": (0, q), "k": (q, kv), "v": (q + kv, kv)}[shard_id]
        param.data.narrow(0, offset, size).copy_(loaded.narrow(0, tp_rank() * size, size))


class RowParallelLinear(LinearBase):
    """Each rank holds input columns [rank * in/tp, ...) and sums the partial outputs."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        assert input_size % tp_size() == 0, f"{input_size} inputs over {tp_size()} ranks"
        super().__init__(input_size // tp_size(), output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if param.dim() == 1:  # the bias is added once, on the summed output
            param.data.copy_(loaded)
            return
        shard = param.shape[1]
        param.data.copy_(loaded.narrow(1, tp_rank() * shard, shard))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = all_reduce(F.linear(x, self.weight))
        return out if self.bias is None else out + self.bias
