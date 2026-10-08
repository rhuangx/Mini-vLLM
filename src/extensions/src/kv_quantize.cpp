#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

void kv_quantize_scatter(at::Tensor pool, const at::Tensor& slot_mapping, const at::Tensor& values, double scale) {
  check_cuda(pool, "pool");
  check_cuda(slot_mapping, "slot_mapping");
  check_cuda(values, "values");
  check_floating(values, "values");
  check_int32(slot_mapping, "slot_mapping");
  TORCH_CHECK(pool.scalar_type() == at::kByte, "fp8 pool must be uint8");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == values.size(0),
              "slot_mapping must be one int32 per token");

  const at::Tensor slots = slot_mapping.contiguous();
  const at::Tensor source = values.contiguous();
  launch_kv_quantize_scatter(pool, slots, source, static_cast<float>(scale));
}

}  // namespace mini_vllm_ext
