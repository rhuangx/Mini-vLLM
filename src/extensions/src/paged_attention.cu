#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cooperative_matrix.h"
#include "mini_vllm_ext.h"

namespace {

using mini_vllm::warp_all_reduce_max;
using mini_vllm::warp_all_reduce_sum;

constexpr int kThreads = 256;
constexpr int kWarps = 32;
constexpr int kBlock = kWarps * 32;
constexpr int kMaxPerLane = 4;

template <typename scalar_t>
__global__ void paged_cache_update_kernel(const int* __restrict__ slot_mapping, const scalar_t* __restrict__ values,
                                          scalar_t* __restrict__ pool, int row, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int slot = slot_mapping[index / row];
  if (slot < 0) {  // a padded row of a captured batch: nothing to write
    return;
  }
  pool[static_cast<size_t>(slot) * row + index % row] = values[index];
}

template <typename T>
__device__ float load_page(const T* pages, size_t index, float) {
  return static_cast<float>(pages[index]);
}

template <>
__device__ float load_page(const uint8_t* pages, size_t index, float scale) {
  __nv_fp8_e4m3 bits;
  bits.__x = pages[index];
  return static_cast<float>(bits) * scale;
}

// One block per (token, query head). 32 warps walk every 32nd visible key with a base-2
// online softmax; each lane owns a contiguous slice of the head. The partials then merge.
template <typename scalar_t, typename page_t>
__global__ void paged_attention_kernel(const scalar_t* __restrict__ q, const page_t* __restrict__ key_pages,
                                       const page_t* __restrict__ value_pages, const int* __restrict__ block_tables,
                                       const int* __restrict__ cu_seqlens_q, const int* __restrict__ context_lens,
                                       scalar_t* __restrict__ out, int num_heads, int num_kv_heads, int head_dim,
                                       int block_size, int num_sequences, int max_blocks, float scale_log2,
                                       float k_scale, float v_scale) {
  const int token = blockIdx.x / num_heads;
  const int head = blockIdx.x - token * num_heads;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int per_lane = (head_dim + 31) / 32;

  int low = 0;
  int high = num_sequences - 1;
  while (low < high) {
    const int mid = (low + high + 1) / 2;
    if (cu_seqlens_q[mid] <= token) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  const int sequence = low;
  const int query_len = cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence];
  const int query_position = token - cu_seqlens_q[sequence];
  const int context = context_lens[sequence];
  const int visible = context - query_len + query_position + 1;
  const int* table = block_tables + sequence * max_blocks;

  float query[kMaxPerLane] = {};
  float accumulator[kMaxPerLane] = {};
  for (int item = 0; item < per_lane; ++item) {
    const int dim = lane * per_lane + item;
    if (dim < head_dim) {
      const size_t q_index = (static_cast<size_t>(token) * num_heads + head) * head_dim + dim;
      query[item] = static_cast<float>(q[q_index]) * scale_log2;
    }
  }

  float max_score = -INFINITY;
  float sum = 0.0f;
  const int row = num_kv_heads * head_dim;
  for (int position = warp; position < visible; position += kWarps) {
    const int slot = table[position / block_size] * block_size + position % block_size;
    const size_t base = static_cast<size_t>(slot) * row + kv_head * head_dim;
    float partial = 0.0f;
    for (int item = 0; item < per_lane; ++item) {
      const int dim = lane * per_lane + item;
      if (dim < head_dim) {
        partial += query[item] * load_page(key_pages, base + dim, k_scale);
      }
    }
    const float score = warp_all_reduce_sum(partial);
    const float new_max = fmaxf(max_score, score);
    const float old_factor = exp2f(max_score - new_max);
    const float factor = exp2f(score - new_max);
    sum = sum * old_factor + factor;
    for (int item = 0; item < per_lane; ++item) {
      const int dim = lane * per_lane + item;
      if (dim < head_dim) {
        accumulator[item] = accumulator[item] * old_factor + factor * load_page(value_pages, base + dim, v_scale);
      }
    }
    max_score = new_max;
  }

  extern __shared__ float scratch[];
  float* outputs = scratch;
  float* maxima = outputs + kWarps * 32;
  float* sums = maxima + kWarps;
  if (lane == 0) {
    maxima[warp] = max_score;
    sums[warp] = sum;
  }
  __syncthreads();
  const float group_max = maxima[lane];
  const float global_max = warp_all_reduce_max(group_max);
  const float factor = group_max == -INFINITY ? 0.0f : exp2f(group_max - global_max);
  const float global_sum = warp_all_reduce_sum(sums[lane] * factor);

  for (int item = 0; item < per_lane; ++item) {
    outputs[lane * kWarps + warp] = accumulator[item];
    __syncthreads();
    accumulator[item] = warp_all_reduce_sum(outputs[warp * kWarps + lane] * factor);
    __syncthreads();
  }
  if (lane == 0) {
    for (int item = 0; item < per_lane; ++item) {
      const int dim = warp * per_lane + item;
      if (dim < head_dim) {
        const size_t index = (static_cast<size_t>(token) * num_heads + head) * head_dim + dim;
        out[index] = static_cast<scalar_t>(global_sum == 0.0f ? 0.0f : accumulator[item] / global_sum);
      }
    }
  }
}

}  // namespace

namespace mini_vllm_ext {

void launch_paged_cache_update(at::Tensor& pool, const at::Tensor& slot_mapping, const at::Tensor& values) {
  const int row = values.numel() / values.size(0);
  const int64_t count = values.numel();
  if (count == 0) {
    return;
  }
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, values.scalar_type(), "paged_cache_update", [&] {
    paged_cache_update_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        slot_mapping.data_ptr<int>(), values.data_ptr<scalar_t>(), pool.data_ptr<scalar_t>(), row, count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_paged_attention(const at::Tensor& q, const at::Tensor& key_pages, const at::Tensor& value_pages,
                            const at::Tensor& block_tables, const at::Tensor& cu_seqlens_q,
                            const at::Tensor& context_lens, at::Tensor& out, float scale, float k_scale,
                            float v_scale) {
  const int tokens = q.size(0);
  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int block_size = key_pages.size(1);
  const int num_kv_heads = key_pages.size(2);
  const int num_sequences = context_lens.size(0);
  const int max_blocks = block_tables.size(1);
  const float scale_log2 = scale * 1.4426950408889634f;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared = (static_cast<size_t>(kWarps) * 32 + 2 * kWarps) * sizeof(float);
  const bool fp8 = key_pages.scalar_type() == at::kByte;

  auto launch = [&](auto page_tag) {
    using page_t = decltype(page_tag);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "paged_attention", [&] {
      paged_attention_kernel<scalar_t, page_t><<<tokens * num_heads, kBlock, shared, stream>>>(
          q.data_ptr<scalar_t>(), key_pages.data_ptr<page_t>(), value_pages.data_ptr<page_t>(),
          block_tables.data_ptr<int>(), cu_seqlens_q.data_ptr<int>(), context_lens.data_ptr<int>(),
          out.data_ptr<scalar_t>(), num_heads, num_kv_heads, head_dim, block_size, num_sequences, max_blocks,
          scale_log2, k_scale, v_scale);
    });
  };
  if (fp8) {
    launch(uint8_t{});
  } else if (q.scalar_type() == at::kFloat) {
    launch(float{});
  } else if (q.scalar_type() == at::kHalf) {
    launch(at::Half{});
  } else {
    launch(at::BFloat16{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mini_vllm_ext
