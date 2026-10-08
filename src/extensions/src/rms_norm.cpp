#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  check_cuda(x, "x");
  check_cuda(weight, "weight");
  check_floating(x, "x");
  check_same_dtype(x, weight, "x", "weight");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  TORCH_CHECK(x.size(-1) == weight.size(0), "weight length must match x's last dimension");

  const at::Tensor input = x.contiguous();
  const at::Tensor weights = weight.contiguous();
  at::Tensor out = at::empty(input.sizes(), input.options());
  if (input.numel() == 0) {
    return out;
  }
  launch_rms_norm(input, weights, out, static_cast<float>(eps));
  return out;
}

}  // namespace mini_vllm_ext
