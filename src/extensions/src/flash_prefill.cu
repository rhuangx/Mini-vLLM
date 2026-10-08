#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cooperative_matrix.h"
#include "mini_vllm_ext.h"

namespace {

using namespace mini_vllm;

constexpr int kHeadDim = 128;
constexpr int kQueryBlock = 64;
constexpr int kKeyBlock = 16;
constexpr int kThreads = 128;
constexpr int kOutputTiles = kHeadDim / 8;
constexpr float kLog2e = 1.4426950408889634f;

__device__ FragA load_q(const __nv_bfloat16* tile, int warp, int lane, int kk) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row0 = warp * 16 + group;
  const int row1 = row0 + 8;
  const int column0 = kk + thread * 2;
  const int column1 = kk + 8 + thread * 2;
  // Register order is (low row, low k), (high row, low k), (low row, high k), (high row, high k).
  FragA fragment;
  fragment.reg[0] = pack_bf16(tile[row0 * kHeadDim + column0], tile[row0 * kHeadDim + column0 + 1]);
  fragment.reg[1] = pack_bf16(tile[row1 * kHeadDim + column0], tile[row1 * kHeadDim + column0 + 1]);
  fragment.reg[2] = pack_bf16(tile[row0 * kHeadDim + column1], tile[row0 * kHeadDim + column1 + 1]);
  fragment.reg[3] = pack_bf16(tile[row1 * kHeadDim + column1], tile[row1 * kHeadDim + column1 + 1]);
  return fragment;
}

__device__ FragB load_k_transpose(const __nv_bfloat16* tile, int lane, int kk, int key_base) {
  const int key = key_base + (lane >> 2);
  const int row = (lane & 3) * 2;
  FragB fragment;
  fragment.reg[0] = pack_bf16(tile[key * kHeadDim + kk + row], tile[key * kHeadDim + kk + row + 1]);
  fragment.reg[1] = pack_bf16(tile[key * kHeadDim + kk + row + 8], tile[key * kHeadDim + kk + row + 9]);
  return fragment;
}

__device__ FragB load_v(const __nv_bfloat16* tile, int lane, int output_tile) {
  const int column = output_tile * 8 + (lane >> 2);
  const int row = (lane & 3) * 2;
  FragB fragment;
  fragment.reg[0] = pack_bf16(tile[row * kHeadDim + column], tile[(row + 1) * kHeadDim + column]);
  fragment.reg[1] = pack_bf16(tile[(row + 8) * kHeadDim + column], tile[(row + 9) * kHeadDim + column]);
  return fragment;
}

// The C fragment already sits in A's register order for one 8-wide half: adjacent columns,
// low row then high row. The two halves become the low-k and high-k pairs of one A fragment.
__device__ FragA scores_to_probabilities(const FragC& left, const FragC& right) {
  FragA fragment;
  fragment.reg[0] = pack_bf16(__float2bfloat16(left.reg[0]), __float2bfloat16(left.reg[1]));
  fragment.reg[1] = pack_bf16(__float2bfloat16(left.reg[2]), __float2bfloat16(left.reg[3]));
  fragment.reg[2] = pack_bf16(__float2bfloat16(right.reg[0]), __float2bfloat16(right.reg[1]));
  fragment.reg[3] = pack_bf16(__float2bfloat16(right.reg[2]), __float2bfloat16(right.reg[3]));
  return fragment;
}

__device__ void mask_scores(FragC& scores, int lane, int query_row, int key_base, int tile_start, int length,
                            int context, bool causal, float scale_log2) {
  const int column = (lane & 3) * 2;
  const int columns[4] = {column, column + 1, column, column + 1};
  const int queries[4] = {query_row, query_row, query_row + 8, query_row + 8};
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    const int key = tile_start + key_base + columns[element];
    const int query = queries[element];
    bool valid = query < length && key < context;
    if (causal) {
      valid = valid && key <= query + (context - length);
    }
    scores.reg[element] = valid ? scores.reg[element] * scale_log2 : -INFINITY;
  }
}

__device__ void online_softmax_row(float& running_max, float& running_sum, float first, float second, float other_first,
                                   float other_second, float& probability_first, float& probability_second,
                                   float& other_probability_first, float& other_probability_second, float& previous) {
  const float tile_max = fmaxf(row_reduce_max(first, second), row_reduce_max(other_first, other_second));
  const float new_max = fmaxf(running_max, tile_max);
  const bool finite = new_max != -INFINITY;
  previous = (!finite || running_max == -INFINITY) ? 0.0f : exp2f(running_max - new_max);
  auto probability = [&](float score) { return (!finite || score == -INFINITY) ? 0.0f : exp2f(score - new_max); };
  probability_first = probability(first);
  probability_second = probability(second);
  other_probability_first = probability(other_first);
  other_probability_second = probability(other_second);
  const float tile_sum =
      row_reduce_sum(probability_first, probability_second) + row_reduce_sum(other_probability_first, other_probability_second);
  running_sum = previous * running_sum + tile_sum;
  running_max = new_max;
}

// bf16, D = 128. One block owns 64 queries of one head; each of its 4 warps owns 16 of them
// and computes them with mma.sync.m16n8k16. Keys stream in tiles of 16. Tiles past the causal
// edge are skipped. The score fragment is reused, after a layout shuffle, as the A fragment of P @ V.
__global__ void __launch_bounds__(kThreads) flash_prefill_kernel(const __nv_bfloat16* __restrict__ q,
                                                                 const __nv_bfloat16* __restrict__ k,
                                                                 const __nv_bfloat16* __restrict__ v,
                                                                 __nv_bfloat16* __restrict__ out, int length,
                                                                 int context, float scale, int causal, int num_heads,
                                                                 int num_kv_heads) {
  const int query_block = blockIdx.x;
  const int query_row = blockIdx.y;
  const int batch = query_row / num_heads;
  const int head = query_row - batch * num_heads;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int kv_row = batch * num_kv_heads + kv_head;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int query_origin = query_block * kQueryBlock;
  const int live_queries = max(0, min(kQueryBlock, length - query_origin));
  const float scale_log2 = scale * kLog2e;
  const int local_query = warp * 16 + (lane >> 2);

  __shared__ __nv_bfloat16 q_tile[kQueryBlock * kHeadDim];
  __shared__ __nv_bfloat16 k_tile[kKeyBlock * kHeadDim];
  __shared__ __nv_bfloat16 v_tile[kKeyBlock * kHeadDim];

  const __nv_bfloat16* q_source = q + (static_cast<size_t>(query_row) * length + query_origin) * kHeadDim;
  load_tile(q_source, kHeadDim, q_tile, kHeadDim, kQueryBlock, kHeadDim, live_queries, kHeadDim);
  __syncthreads();

  FragC output[kOutputTiles];
#pragma unroll
  for (int tile = 0; tile < kOutputTiles; ++tile) {
    output[tile] = FragC{};
  }
  float max_low = -INFINITY;
  float max_high = -INFINITY;
  float sum_low = 0.0f;
  float sum_high = 0.0f;

  const int total_tiles = (context + kKeyBlock - 1) / kKeyBlock;
  int tile_limit = total_tiles;
  if (causal) {
    const int last_query = min((query_block + 1) * kQueryBlock, length) - 1;
    const int last_key = last_query + (context - length);
    tile_limit = min(total_tiles, max(0, (last_key + kKeyBlock) / kKeyBlock));
  }

  for (int tile = 0; tile < tile_limit; ++tile) {
    const int tile_start = tile * kKeyBlock;
    const int live_keys = max(0, min(kKeyBlock, context - tile_start));
    const __nv_bfloat16* k_source = k + (static_cast<size_t>(kv_row) * context + tile_start) * kHeadDim;
    load_tile(k_source, kHeadDim, k_tile, kHeadDim, kKeyBlock, kHeadDim, live_keys, kHeadDim);
    __syncthreads();

    FragC left{};
    FragC right{};
    for (int kk = 0; kk < kHeadDim; kk += 16) {
      const FragA query = load_q(q_tile, warp, lane, kk);
      mma_m16n8k16(left, query, load_k_transpose(k_tile, lane, kk, 0));
      mma_m16n8k16(right, query, load_k_transpose(k_tile, lane, kk, 8));
    }
    mask_scores(left, lane, query_origin + local_query, 0, tile_start, length, context, causal != 0, scale_log2);
    mask_scores(right, lane, query_origin + local_query, 8, tile_start, length, context, causal != 0, scale_log2);

    float previous_low = 0.0f;
    float previous_high = 0.0f;
    online_softmax_row(max_low, sum_low, left.reg[0], left.reg[1], right.reg[0], right.reg[1], left.reg[0], left.reg[1],
                       right.reg[0], right.reg[1], previous_low);
    online_softmax_row(max_high, sum_high, left.reg[2], left.reg[3], right.reg[2], right.reg[3], left.reg[2],
                       left.reg[3], right.reg[2], right.reg[3], previous_high);

    const FragA probabilities = scores_to_probabilities(left, right);
    __syncthreads();
    const __nv_bfloat16* v_source = v + (static_cast<size_t>(kv_row) * context + tile_start) * kHeadDim;
    load_tile(v_source, kHeadDim, v_tile, kHeadDim, kKeyBlock, kHeadDim, live_keys, kHeadDim);
    __syncthreads();

#pragma unroll
    for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
      output[output_tile].reg[0] *= previous_low;
      output[output_tile].reg[1] *= previous_low;
      output[output_tile].reg[2] *= previous_high;
      output[output_tile].reg[3] *= previous_high;
      mma_m16n8k16(output[output_tile], probabilities, load_v(v_tile, lane, output_tile));
    }
    __syncthreads();
  }

  const int query_low = query_origin + local_query;
  const int query_high = query_low + 8;
  const float inverse_low = sum_low == 0.0f ? 0.0f : 1.0f / sum_low;
  const float inverse_high = sum_high == 0.0f ? 0.0f : 1.0f / sum_high;
#pragma unroll
  for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    const int column = output_tile * 8 + (lane & 3) * 2;
    if (query_low < length) {
      out[(static_cast<size_t>(query_row) * length + query_low) * kHeadDim + column] =
          __float2bfloat16(output[output_tile].reg[0] * inverse_low);
      out[(static_cast<size_t>(query_row) * length + query_low) * kHeadDim + column + 1] =
          __float2bfloat16(output[output_tile].reg[1] * inverse_low);
    }
    if (query_high < length) {
      out[(static_cast<size_t>(query_row) * length + query_high) * kHeadDim + column] =
          __float2bfloat16(output[output_tile].reg[2] * inverse_high);
      out[(static_cast<size_t>(query_row) * length + query_high) * kHeadDim + column + 1] =
          __float2bfloat16(output[output_tile].reg[3] * inverse_high);
    }
  }
}

}  // namespace

namespace mini_vllm_ext {

void launch_flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, float scale,
                          bool causal, int num_heads, int num_kv_heads) {
  const int length = q.size(-2);
  const int context = k.size(-2);
  const int query_rows = q.numel() / (length * kHeadDim);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((length + kQueryBlock - 1) / kQueryBlock, query_rows);
  flash_prefill_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), length,
      context, scale, causal ? 1 : 0, num_heads, num_kv_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mini_vllm_ext
