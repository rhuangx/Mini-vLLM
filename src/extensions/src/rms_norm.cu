#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cooperative_matrix.h"
#include "mini_vllm_ext.h"

namespace {

using namespace mini_vllm;

// One block per row. The normalized value is rounded to T before the weight multiply.
template <typename scalar_t, typename chunk_t>
__global__ void rms_norm_kernel(const scalar_t* __restrict__ input, const scalar_t* __restrict__ weight,
                                scalar_t* __restrict__ out, int64_t dim, float eps) {
  constexpr int kLanes = chunk_t::kLanes;
  const int64_t row = blockIdx.x;
  const int64_t chunks = dim / kLanes;
  const chunk_t* row_in = reinterpret_cast<const chunk_t*>(input + row * dim);
  chunk_t* row_out = reinterpret_cast<chunk_t*>(out + row * dim);
  const chunk_t* row_weight = reinterpret_cast<const chunk_t*>(weight);

  const bool resident = chunks <= static_cast<int64_t>(blockDim.x);
  chunk_t held;

  float sum_squares = 0.0f;
  for (int64_t i = threadIdx.x; i < chunks; i += blockDim.x) {
    const chunk_t chunk = row_in[i];
    if (resident) {
      held = chunk;
    }
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      const float value = static_cast<float>(chunk.lane[j]);
      sum_squares += value * value;
    }
  }

  __shared__ float warp_totals[kMaxThreads / kWarpSize];
  __shared__ float inverse_rms;
  const float total = block_reduce_sum(sum_squares, warp_totals);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(total / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < chunks; i += blockDim.x) {
    const chunk_t chunk = resident ? held : row_in[i];
    const chunk_t weights = row_weight[i];
    chunk_t result;
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      const scalar_t normalized = static_cast<scalar_t>(static_cast<float>(chunk.lane[j]) * inverse_rms);
      result.lane[j] = static_cast<scalar_t>(static_cast<float>(normalized) * static_cast<float>(weights.lane[j]));
    }
    row_out[i] = result;
  }
}

template <typename scalar_t>
void launch_typed(const at::Tensor& input, const at::Tensor& weight, at::Tensor& out, float eps) {
  const int64_t dim = input.size(-1);
  const int64_t rows = input.numel() / dim;
  const scalar_t* input_pointer = input.data_ptr<scalar_t>();
  const scalar_t* weight_pointer = weight.data_ptr<scalar_t>();
  scalar_t* out_pointer = out.data_ptr<scalar_t>();
  constexpr int kLanes = Vector<scalar_t>::kLanes;
  const bool vectorizable = dim % kLanes == 0 && is_vector_aligned(input_pointer) && is_vector_aligned(weight_pointer) &&
                            is_vector_aligned(out_pointer);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vectorizable) {
    rms_norm_kernel<scalar_t, Vector<scalar_t>>
        <<<rows, threads_for(dim / kLanes), 0, stream>>>(input_pointer, weight_pointer, out_pointer, dim, eps);
  } else {
    rms_norm_kernel<scalar_t, Scalar<scalar_t>>
        <<<rows, threads_for(dim), 0, stream>>>(input_pointer, weight_pointer, out_pointer, dim, eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace mini_vllm_ext {

void launch_rms_norm(const at::Tensor& input, const at::Tensor& weight, at::Tensor& out, float eps) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "rms_norm", [&] {
    launch_typed<scalar_t>(input, weight, out, eps);
  });
}

}  // namespace mini_vllm_ext
