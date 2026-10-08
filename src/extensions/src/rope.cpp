#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor rope(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos, const at::Tensor& sin) {
  check_cuda(x, "x");
  check_cuda(positions, "positions");
  check_cuda(cos, "cos");
  check_cuda(sin, "sin");
  check_floating(x, "x");
  TORCH_CHECK(x.dim() == 4, "rope x must be [B, L, H, D]");
  TORCH_CHECK(x.size(-1) % 2 == 0, "rope head dim must be even");
  TORCH_CHECK(positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
              "positions must be int32 or int64");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat, "cos and sin tables must be fp32");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2, "positions must be [L] or [B, L]");

  const at::Tensor input = x.contiguous();
  const at::Tensor position_ids = positions.to(at::kInt).contiguous();
  const at::Tensor cos_table = cos.contiguous();
  const at::Tensor sin_table = sin.contiguous();
  at::Tensor out = at::empty(input.sizes(), input.options());
  if (input.numel() == 0) {
    return out;
  }
  launch_rope(input, position_ids, cos_table, sin_table, out);
  return out;
}

}  // namespace mini_vllm_ext
