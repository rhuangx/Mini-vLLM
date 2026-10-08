#include "mini_vllm_ext.h"

namespace {

TORCH_LIBRARY(mini_vllm_ext, m) {
  m.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
  m.def("swiglu(Tensor gate, Tensor up) -> Tensor");
  m.def("rope(Tensor x, Tensor positions, Tensor cos, Tensor sin) -> Tensor");
  m.def("paged_cache_update(Tensor(a!) pool, Tensor slot_mapping, Tensor values) -> ()");
  m.def("paged_attention(Tensor q, Tensor key_pages, Tensor value_pages, Tensor block_tables, Tensor cu_seqlens_q, "
        "Tensor context_lens, float scale, float k_scale, float v_scale) -> Tensor");
  m.def("decode_attention(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor");
  m.def("flash_prefill(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor");
  m.def("kv_quantize_scatter(Tensor(a!) pool, Tensor slot_mapping, Tensor values, float scale) -> ()");
}

TORCH_LIBRARY_IMPL(mini_vllm_ext, CUDA, m) {
  m.impl("rms_norm", mini_vllm_ext::rms_norm);
  m.impl("swiglu", mini_vllm_ext::swiglu);
  m.impl("rope", mini_vllm_ext::rope);
  m.impl("paged_cache_update", mini_vllm_ext::paged_cache_update);
  m.impl("paged_attention", mini_vllm_ext::paged_attention);
  m.impl("decode_attention", mini_vllm_ext::decode_attention);
  m.impl("flash_prefill", mini_vllm_ext::flash_prefill);
  m.impl("kv_quantize_scatter", mini_vllm_ext::kv_quantize_scatter);
}

}  // namespace
