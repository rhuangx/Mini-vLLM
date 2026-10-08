#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cooperative_matrix.h"
#include "mini_vllm_ext.h"

namespace {

using mini_vllm::warp_all_reduce_max;
using mini_vllm::warp_all_reduce_sum;

constexpr int kWarps = 32;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxPerLane = 8;

// One block per query token of one head. 32 warps split the context; each lane owns
// every 32nd dimension. Online softmax in natural exp, then the partials merge in shared memory.
template <typename scalar_t>
__global__ void decode_attention_kernel(const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
                                        const scalar_t* __restrict__ v, scalar_t* __restrict__ out, int length,
                                        int context, int dim, int num_heads, int num_kv_heads, float scale,
                                        int causal) {
  const int query_index = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int query_row = query_index / length;
  const int query_position = query_index - query_row * length;
  const int batch = query_row / num_heads;
  const int kv_head = (query_row - batch * num_heads) / (num_heads / num_kv_heads);
  const int kv_row = batch * num_kv_heads + kv_head;
  const int per_lane = (dim + 31) / 32;
  const int visible = causal ? context - length + query_position + 1 : context;

  float query[kMaxPerLane] = {};
  float accumulator[kMaxPerLane] = {};
  for (int item = 0; item < per_lane; ++item) {
    const int d = lane + item * 32;
    if (d < dim) {
      query[item] = static_cast<float>(q[static_cast<size_t>(query_index) * dim + d]) * scale;
    }
  }

  float max_score = -1e30f;
  float sum = 0.0f;
  for (int position = warp; position < visible; position += kWarps) {
    float partial = 0.0f;
    const size_t base = (static_cast<size_t>(kv_row) * context + position) * dim;
    for (int item = 0; item < per_lane; ++item) {
      const int d = lane + item * 32;
      if (d < dim) {
        partial += query[item] * static_cast<float>(k[base + d]);
      }
    }
    const float score = warp_all_reduce_sum(partial);
    const float new_max = fmaxf(max_score, score);
    const float old_factor = __expf(max_score - new_max);
    const float factor = __expf(score - new_max);
    sum = sum * old_factor + factor;
    for (int item = 0; item < per_lane; ++item) {
      const int d = lane + item * 32;
      if (d < dim) {
        accumulator[item] = accumulator[item] * old_factor + factor * static_cast<float>(v[base + d]);
      }
    }
    max_score = new_max;
  }

  extern __shared__ float scratch[];
  float* partial_values = scratch;
  float* partial_maxima = partial_values + kWarps * dim;
  float* partial_sums = partial_maxima + kWarps;
  float* factors = partial_sums + kWarps;
  if (lane == 0) {
    partial_maxima[warp] = max_score;
    partial_sums[warp] = sum;
  }
  for (int item = 0; item < per_lane; ++item) {
    const int d = lane + item * 32;
    if (d < dim) {
      partial_values[warp * dim + d] = accumulator[item];
    }
  }
  __syncthreads();

  if (warp == 0) {
    const float global_max = warp_all_reduce_max(partial_maxima[lane]);
    factors[lane] = __expf(partial_maxima[lane] - global_max);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float global_sum = 0.0f;
    for (int group = 0; group < kWarps; ++group) {
      global_sum += partial_sums[group] * factors[group];
    }
    partial_sums[0] = global_sum;
  }
  __syncthreads();
  if (threadIdx.x < static_cast<unsigned>(dim)) {
    float value = 0.0f;
    for (int group = 0; group < kWarps; ++group) {
      value += partial_values[group * dim + threadIdx.x] * factors[group];
    }
    const float denom = partial_sums[0];
    out[static_cast<size_t>(query_index) * dim + threadIdx.x] =
        static_cast<scalar_t>(denom == 0.0f ? 0.0f : value / denom);
  }
}

}  // namespace

namespace mini_vllm_ext {

void launch_decode_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                             float scale, bool causal, int num_heads, int num_kv_heads) {
  const int dim = q.size(-1);
  const int length = q.size(-2);
  const int context = k.size(-2);
  const int query_rows = q.numel() / (length * dim);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared = static_cast<size_t>(kWarps) * (dim + 3) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "decode_attention", [&] {
    decode_attention_kernel<scalar_t><<<query_rows * length, kThreads, shared, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), length,
        context, dim, num_heads, num_kv_heads, scale, causal ? 1 : 0);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mini_vllm_ext
