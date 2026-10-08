#pragma once

// Device helpers for the CUDA kernels: warp and block reductions, a global-to-shared
// tile load, and the m16n8k16 bf16 fragment helpers flash_prefill is built on.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace mini_vllm {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;
constexpr int kBytesPerVector = 16;

template <typename scalar_t>
struct alignas(kBytesPerVector) Vector {
  static constexpr int kLanes = kBytesPerVector / sizeof(scalar_t);
  scalar_t lane[kLanes];
};

template <typename scalar_t>
struct Scalar {
  static constexpr int kLanes = 1;
  scalar_t lane[1];
};

__host__ __device__ inline bool is_vector_aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kBytesPerVector == 0;
}

inline int threads_for(int64_t units) {
  const int64_t rounded = ((units + kWarpSize - 1) / kWarpSize) * kWarpSize;
  const int64_t clamped = rounded < kWarpSize ? kWarpSize : rounded;
  return static_cast<int>(clamped > kMaxThreads ? kMaxThreads : clamped);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_all_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_all_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// One float of scratch per warp. Result in thread 0. __syncthreads() between calls.
__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < static_cast<unsigned>(warps)) ? scratch[threadIdx.x] : 0.0f;
  return warp == 0 ? warp_reduce_sum(value) : 0.0f;
}

__device__ __forceinline__ float block_reduce_max(float value, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < static_cast<unsigned>(warps)) ? scratch[threadIdx.x] : -INFINITY;
  return warp == 0 ? warp_reduce_max(value) : -INFINITY;
}

// Copy a row-major tile into shared memory. Out-of-range elements are zero, so a later
// full-tile matrix load never reads a partial edge as data.
template <typename T>
__device__ void load_tile(const T* source, int source_stride, T* destination, int destination_stride,
                          int rows, int columns, int valid_rows, int valid_columns) {
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int total = rows * columns;
  for (int index = tid; index < total; index += threads) {
    const int row = index / columns;
    const int column = index - row * columns;
    T value = T(0);
    if (row < valid_rows && column < valid_columns) {
      value = source[row * source_stride + column];
    }
    destination[row * destination_stride + column] = value;
  }
}

// m16n8k16 fragment coordinates, from the PTX ISA. A is 16x16 row-major bf16 (4 .b32
// registers, two bf16 each). B is 16x8 column-major bf16 (2 .b32). C is 16x8 fp32.
struct FragA {
  unsigned reg[4];
};
struct FragB {
  unsigned reg[2];
};
struct FragC {
  float reg[4];
};

__device__ __forceinline__ unsigned pack_bf16(__nv_bfloat16 low, __nv_bfloat16 high) {
  unsigned out;
  // The first element of the pair lives in the low 16 bits, matching mma's .b32 layout.
  asm volatile("mov.b32 %0, {%1, %2};" : "=r"(out) : "h"(__bfloat16_as_ushort(low)), "h"(__bfloat16_as_ushort(high)));
  return out;
}

__device__ __forceinline__ __nv_bfloat16 unpack_low(unsigned packed) {
  return __ushort_as_bfloat16(static_cast<unsigned short>(packed));
}

__device__ __forceinline__ void mma_m16n8k16(FragC& accumulator, const FragA& a, const FragB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(accumulator.reg[0]), "=f"(accumulator.reg[1]), "=f"(accumulator.reg[2]), "=f"(accumulator.reg[3])
      : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]), "r"(b.reg[0]), "r"(b.reg[1]), "f"(accumulator.reg[0]),
        "f"(accumulator.reg[1]), "f"(accumulator.reg[2]), "f"(accumulator.reg[3]));
}

// Row max/sum of one 16x8 C fragment. A row's 8 values sit on 4 lanes, two each, so
// shfl_xor 1 then 2 covers the row and stays inside that group of four lanes.
__device__ __forceinline__ float row_reduce_max(float first, float second) {
  float value = fmaxf(first, second);
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

__device__ __forceinline__ float row_reduce_sum(float first, float second) {
  float value = first + second;
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value + __shfl_xor_sync(0xffffffffu, value, 2);
}

// ldmatrix.x4: four 8x8 bf16 tiles, the width of one A fragment, from shared memory.
// `address` is the per-lane shared pointer ldmatrix expects (one 8-element row).
__device__ __forceinline__ void ldmatrix_x4(FragA& fragment, unsigned address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(fragment.reg[0]), "=r"(fragment.reg[1]), "=r"(fragment.reg[2]), "=r"(fragment.reg[3])
               : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2(FragB& fragment, unsigned address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(fragment.reg[0]), "=r"(fragment.reg[1])
               : "r"(address));
}

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
  unsigned address;
  asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(address) : "l"(pointer));
  return address;
}

}  // namespace mini_vllm
