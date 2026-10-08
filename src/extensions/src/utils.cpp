#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

void check_cuda(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_floating(const at::Tensor& tensor, const char* name) {
  const auto type = tensor.scalar_type();
  TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16, name,
              " must be fp32, fp16 or bf16, got ", type);
}

void check_same_dtype(const at::Tensor& left, const at::Tensor& right, const char* left_name, const char* right_name) {
  TORCH_CHECK(left.scalar_type() == right.scalar_type(), left_name, " is ", left.scalar_type(), " but ", right_name,
              " is ", right.scalar_type());
}

void check_int32(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32, got ", tensor.scalar_type());
}

void check_dim(const at::Tensor& tensor, int64_t limit, const char* name) {
  TORCH_CHECK(tensor.size(-1) <= limit, name, " head dim ", tensor.size(-1), " exceeds ", limit);
}

}  // namespace mini_vllm_ext
