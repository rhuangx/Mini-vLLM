#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp8.h>

#include "mini_vllm_ext.h"

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void kv_quantize_scatter_kernel(const int* __restrict__ slot_mapping, const scalar_t* __restrict__ values,
                                           uint8_t* __restrict__ pool, int row, float scale, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int slot = slot_mapping[index / row];
  if (slot < 0) {  // a padded row of a captured batch: nothing to write
    return;
  }
  const float scaled = static_cast<float>(values[index]) / scale;
  pool[static_cast<size_t>(slot) * row + index % row] =
      __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace

namespace mini_vllm_ext {

void launch_kv_quantize_scatter(at::Tensor& pool, const at::Tensor& slot_mapping, const at::Tensor& values,
                                float scale) {
  const int row = values.numel() / values.size(0);
  const int64_t count = values.numel();
  if (count == 0) {
    return;
  }
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, values.scalar_type(), "kv_quantize_scatter", [&] {
    kv_quantize_scatter_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        slot_mapping.data_ptr<int>(), values.data_ptr<scalar_t>(), pool.data_ptr<uint8_t>(), row, scale, count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mini_vllm_ext
