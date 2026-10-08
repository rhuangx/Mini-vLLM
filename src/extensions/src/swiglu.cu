#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cooperative_matrix.h"
#include "mini_vllm_ext.h"

namespace {

using namespace mini_vllm;

constexpr int kThreads = 256;

// silu(gate) * up in fp32, rounded once, matching the pure path's single cast.
template <typename scalar_t, typename chunk_t>
__global__ void swiglu_kernel(const scalar_t* __restrict__ gate, const scalar_t* __restrict__ up,
                              scalar_t* __restrict__ out, int64_t chunks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= chunks) {
    return;
  }
  constexpr int kLanes = chunk_t::kLanes;
  const chunk_t gate_chunk = reinterpret_cast<const chunk_t*>(gate)[index];
  const chunk_t up_chunk = reinterpret_cast<const chunk_t*>(up)[index];
  chunk_t result;
#pragma unroll
  for (int j = 0; j < kLanes; ++j) {
    const float g = static_cast<float>(gate_chunk.lane[j]);
    result.lane[j] = static_cast<scalar_t>(g / (1.0f + expf(-g)) * static_cast<float>(up_chunk.lane[j]));
  }
  reinterpret_cast<chunk_t*>(out)[index] = result;
}

template <typename scalar_t>
void launch_typed(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out) {
  const scalar_t* gate_pointer = gate.data_ptr<scalar_t>();
  const scalar_t* up_pointer = up.data_ptr<scalar_t>();
  scalar_t* out_pointer = out.data_ptr<scalar_t>();
  constexpr int kLanes = Vector<scalar_t>::kLanes;
  const int64_t elements = gate.numel();
  const bool vectorizable = elements % kLanes == 0 && is_vector_aligned(gate_pointer) && is_vector_aligned(up_pointer) &&
                            is_vector_aligned(out_pointer);
  const int64_t chunks = vectorizable ? elements / kLanes : elements;
  const int64_t blocks = (chunks + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vectorizable) {
    swiglu_kernel<scalar_t, Vector<scalar_t>>
        <<<blocks, kThreads, 0, stream>>>(gate_pointer, up_pointer, out_pointer, chunks);
  } else {
    swiglu_kernel<scalar_t, Scalar<scalar_t>>
        <<<blocks, kThreads, 0, stream>>>(gate_pointer, up_pointer, out_pointer, chunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace mini_vllm_ext {

void launch_swiglu(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gate.scalar_type(), "swiglu", [&] {
    launch_typed<scalar_t>(gate, up, out);
  });
}

}  // namespace mini_vllm_ext
