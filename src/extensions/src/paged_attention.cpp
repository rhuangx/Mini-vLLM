#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

void paged_cache_update(at::Tensor pool, const at::Tensor& slot_mapping, const at::Tensor& values) {
  check_cuda(pool, "pool");
  check_cuda(slot_mapping, "slot_mapping");
  check_cuda(values, "values");
  check_floating(values, "values");
  check_same_dtype(pool, values, "pool", "values");
  check_int32(slot_mapping, "slot_mapping");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == values.size(0),
              "slot_mapping must be one int32 per token");

  const at::Tensor slots = slot_mapping.contiguous();
  const at::Tensor source = values.contiguous();
  launch_paged_cache_update(pool, slots, source);
}

at::Tensor paged_attention(const at::Tensor& q, const at::Tensor& key_pages, const at::Tensor& value_pages,
                           const at::Tensor& block_tables, const at::Tensor& cu_seqlens_q,
                           const at::Tensor& context_lens, double scale, double k_scale, double v_scale) {
  check_cuda(q, "q");
  check_cuda(key_pages, "key_pages");
  check_cuda(value_pages, "value_pages");
  check_floating(q, "q");
  check_dim(q, 128, "paged_attention");
  TORCH_CHECK(q.dim() == 3, "paged_attention q must be [T, H_q, D]");
  check_int32(block_tables, "block_tables");
  check_int32(cu_seqlens_q, "cu_seqlens_q");
  check_int32(context_lens, "context_lens");
  const bool fp8 = key_pages.scalar_type() == at::kByte;
  if (!fp8) {
    check_same_dtype(q, key_pages, "q", "key_pages");
  }
  check_same_dtype(key_pages, value_pages, "key_pages", "value_pages");

  const at::Tensor query = q.contiguous();
  const at::Tensor keys = key_pages.contiguous();
  const at::Tensor values = value_pages.contiguous();
  const at::Tensor tables = block_tables.contiguous();
  const at::Tensor starts = cu_seqlens_q.contiguous();
  const at::Tensor lengths = context_lens.contiguous();
  at::Tensor out = at::empty(query.sizes(), query.options());
  launch_paged_attention(query, keys, values, tables, starts, lengths, out, static_cast<float>(scale),
                         static_cast<float>(k_scale), static_cast<float>(v_scale));
  return out;
}

}  // namespace mini_vllm_ext
