#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "mini_vllm_ext.h"

namespace {

constexpr int kThreads = 256;

// silu(gate) * up in fp32, rounded once, where each row of x is [gate | up]: the merged
// gate_up projection's output, read in place instead of copied into two tensors.
template <typename scalar_t>
__global__ void silu_and_mul_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int64_t width,
                                    int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int64_t row = index / width;
  const int64_t col = index % width;
  const float gate = static_cast<float>(x[row * 2 * width + col]);
  const float up = static_cast<float>(x[row * 2 * width + width + col]);
  out[index] = static_cast<scalar_t>(gate / (1.0f + expf(-gate)) * up);
}

}  // namespace

namespace mini_vllm_ext {

void launch_silu_and_mul(const at::Tensor& x, at::Tensor& out) {
  const int64_t count = out.numel();
  const int64_t width = out.size(-1);
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "silu_and_mul", [&] {
    silu_and_mul_kernel<scalar_t>
        <<<blocks, kThreads, 0, stream>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), width, count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mini_vllm_ext
