#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor silu_and_mul(const at::Tensor& x) {
  check_cuda(x, "x");
  check_floating(x, "x");
  TORCH_CHECK(x.size(-1) % 2 == 0, "the last axis holds gate and up, so it must be even");

  const at::Tensor input = x.contiguous();
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  shape.back() /= 2;
  at::Tensor out = at::empty(shape, input.options());
  if (out.numel() == 0) {
    return out;
  }
  launch_silu_and_mul(input, out);
  return out;
}

}  // namespace mini_vllm_ext
