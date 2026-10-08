#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor decode_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale, bool causal) {
  check_cuda(q, "q");
  check_cuda(k, "k");
  check_cuda(v, "v");
  check_floating(q, "q");
  check_same_dtype(q, k, "q", "k");
  check_same_dtype(q, v, "q", "v");
  TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "decode_attention q must be 3-D or 4-D");
  TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim(), "k and v must match q's rank");
  check_dim(q, 256, "decode_attention");
  TORCH_CHECK(q.size(-2) <= 8, "decode_attention covers at most 8 query tokens, got ", q.size(-2));

  const at::Tensor query = q.contiguous();
  const at::Tensor key = k.contiguous();
  const at::Tensor value = v.contiguous();
  at::Tensor out = at::empty(query.sizes(), query.options());
  const int num_heads = query.size(-3);
  const int num_kv_heads = key.size(-3);
  launch_decode_attention(query, key, value, out, static_cast<float>(scale), causal, num_heads, num_kv_heads);
  return out;
}

}  // namespace mini_vllm_ext
