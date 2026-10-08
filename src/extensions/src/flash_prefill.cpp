#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale, bool causal) {
  check_cuda(q, "q");
  check_cuda(k, "k");
  check_cuda(v, "v");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "flash_prefill is bf16 only, got ", q.scalar_type());
  check_same_dtype(q, k, "q", "k");
  check_same_dtype(q, v, "q", "v");
  TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "flash_prefill q must be 3-D or 4-D");
  TORCH_CHECK(q.size(-1) == 128, "flash_prefill requires D = 128, got ", q.size(-1));

  const at::Tensor query = q.contiguous();
  const at::Tensor key = k.contiguous();
  const at::Tensor value = v.contiguous();
  at::Tensor out = at::empty(query.sizes(), query.options());
  launch_flash_prefill(query, key, value, out, static_cast<float>(scale), causal, query.size(-3), key.size(-3));
  return out;
}

}  // namespace mini_vllm_ext
