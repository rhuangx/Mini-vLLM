#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "mini_vllm_ext.h"

namespace {

constexpr int kThreads = 256;

// One thread per rotated pair. Tables are the fp32 ones positional_encoding.py built.
template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* __restrict__ x, const int* __restrict__ positions,
                            const float* __restrict__ cos_table, const float* __restrict__ sin_table,
                            scalar_t* __restrict__ out, int length, int heads, int head_dim, int position_stride,
                            int64_t pairs) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) {
    return;
  }
  const int half_dim = head_dim / 2;
  const int pair = index % half_dim;
  const int row = index / half_dim;
  const int token = row / heads;
  const int batch = token / length;
  const int position_in_row = token - batch * length;
  const int position = positions[batch * position_stride + position_in_row];

  const int first = row * head_dim + pair;
  const int second = first + half_dim;
  const float x1 = static_cast<float>(x[first]);
  const float x2 = static_cast<float>(x[second]);
  const int table = position * head_dim + pair;
  out[first] = static_cast<scalar_t>(x1 * cos_table[table] - x2 * sin_table[table]);
  out[second] = static_cast<scalar_t>(x2 * cos_table[table + half_dim] + x1 * sin_table[table + half_dim]);
}

template <typename scalar_t>
void launch_typed(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos, const at::Tensor& sin,
                  at::Tensor& out) {
  const int length = x.size(1);
  const int heads = x.size(2);
  const int head_dim = x.size(3);
  const int position_stride = positions.dim() == 1 ? 0 : length;
  const int64_t pairs = x.numel() / 2;
  const int64_t blocks = (pairs + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  rope_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
      x.data_ptr<scalar_t>(), positions.data_ptr<int>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
      out.data_ptr<scalar_t>(), length, heads, head_dim, position_stride, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace mini_vllm_ext {

void launch_rope(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos, const at::Tensor& sin,
                 at::Tensor& out) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rope", [&] {
    launch_typed<scalar_t>(x, positions, cos, sin, out);
  });
}

}  // namespace mini_vllm_ext
