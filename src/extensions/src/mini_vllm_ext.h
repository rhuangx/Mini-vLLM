#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace mini_vllm_ext {

void check_cuda(const at::Tensor& tensor, const char* name);
void check_floating(const at::Tensor& tensor, const char* name);
void check_same_dtype(const at::Tensor& left, const at::Tensor& right, const char* left_name, const char* right_name);
void check_int32(const at::Tensor& tensor, const char* name);
void check_dim(const at::Tensor& tensor, int64_t limit, const char* name);

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps);
at::Tensor silu_and_mul(const at::Tensor& x);
at::Tensor rope(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos, const at::Tensor& sin);
void paged_cache_update(at::Tensor pool, const at::Tensor& slot_mapping, const at::Tensor& values);
void kv_quantize_scatter(at::Tensor pool, const at::Tensor& slot_mapping, const at::Tensor& values, double scale);
at::Tensor decode_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale, bool causal);
at::Tensor flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale, bool causal);
at::Tensor paged_attention(const at::Tensor& q, const at::Tensor& key_pages, const at::Tensor& value_pages,
                           const at::Tensor& block_tables, const at::Tensor& cu_seqlens_q,
                           const at::Tensor& context_lens, double scale, double k_scale, double v_scale);

void launch_rms_norm(const at::Tensor& input, const at::Tensor& weight, at::Tensor& out, float eps);
void launch_silu_and_mul(const at::Tensor& x, at::Tensor& out);
void launch_rope(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos, const at::Tensor& sin,
                 at::Tensor& out);
void launch_paged_cache_update(at::Tensor& pool, const at::Tensor& slot_mapping, const at::Tensor& values);
void launch_kv_quantize_scatter(at::Tensor& pool, const at::Tensor& slot_mapping, const at::Tensor& values, float scale);
void launch_decode_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, float scale,
                             bool causal, int num_heads, int num_kv_heads);
void launch_flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, float scale,
                          bool causal, int num_heads, int num_kv_heads);
void launch_paged_attention(const at::Tensor& q, const at::Tensor& key_pages, const at::Tensor& value_pages,
                            const at::Tensor& block_tables, const at::Tensor& cu_seqlens_q,
                            const at::Tensor& context_lens, at::Tensor& out, float scale, float k_scale, float v_scale);

}  // namespace mini_vllm_ext
