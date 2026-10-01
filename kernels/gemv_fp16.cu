//===----------------------------------------------------------------------===//
// kernels/gemv_fp16.cu -- Dense matrix-vector product in FP16 for decode.
//
// Computes out[N] = x[K] * A[K x N], row-major.
// Equivalent to matmul with M=1:
//   out[j] = sum_{k=0}^{K-1} x[k] * A[k * N + j]
//
// ACCUMULATION AND NUMERICS:
//
// Accumulation is performed strictly in single-precision FP32 (`float acc`)
// to prevent underflow, catastrophic cancellation, or overflow across K=4096
// elements. At the end of reduction, the accumulator is rounded to binary16
// using __float2half / __float22half2_rn.
//
// MEMORY ARCHITECTURE AND TARGET:
//
// For K=4096, N=4096:
//   - Matrix A footprint: 4096 * 4096 * 2 bytes = 32 MiB.
//   - Input x footprint: 4096 * 2 bytes = 8 KiB (resides in L1/Texture cache).
//   - Output out footprint: 4096 * 2 bytes = 8 KiB.
//   - Target latency at ~460 GB/s bandwidth: ~0.070 ms.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

// Fast path for N % 64 == 0 and K % 8 == 0:
// 256 threads per block = 8 warps.
// Each block computes 64 output columns.
// Each lane processes 2 columns using half2 vector memory instructions.
// 8 warps cooperatively reduce over the K dimension via shared memory.
__global__ void gemv_fp16_fast_k8_n64(const half* __restrict__ A,
                                      const half* __restrict__ x, half* __restrict__ out,
                                      int64_t N, int64_t K) {
  constexpr int WARPS = 8;
  constexpr int COLS_PER_BLOCK = 64;

  __shared__ float s_red[WARPS][COLS_PER_BLOCK];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * COLS_PER_BLOCK;

  // Each lane processes 2 columns: lane_id * 2 and lane_id * 2 + 1
  const int64_t c0 = col_base + lane_id * 2;
  const int64_t c1 = c0 + 1;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  if (c1 < N) {
#pragma unroll 4
    for (int64_t k = warp_id; k < K; k += WARPS) {
      const float xk = __half2float(x[k]);
      const half2 a_val = *reinterpret_cast<const half2*>(&A[k * N + c0]);
      const float2 f_a = __half22float2(a_val);
      acc0 = fmaf(xk, f_a.x, acc0);
      acc1 = fmaf(xk, f_a.y, acc1);
    }
  } else if (c0 < N) {
#pragma unroll 4
    for (int64_t k = warp_id; k < K; k += WARPS) {
      const float xk = __half2float(x[k]);
      acc0 = fmaf(xk, __half2float(A[k * N + c0]), acc0);
    }
  }

  s_red[warp_id][lane_id * 2] = acc0;
  s_red[warp_id][lane_id * 2 + 1] = acc1;

  __syncthreads();

  // Warp 0 aggregates partial sums across all 8 warps and writes output
  if (warp_id == 0) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;

#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      sum0 += s_red[w][lane_id * 2];
      sum1 += s_red[w][lane_id * 2 + 1];
    }

    if (c1 < N) {
      *reinterpret_cast<half2*>(&out[c0]) = __float22half2_rn(make_float2(sum0, sum1));
    } else if (c0 < N) {
      out[c0] = __float2half(sum0);
    }
  }
}

// Fallback kernel for arbitrary N and K
__global__ void gemv_fp16_scalar_fallback(const half* __restrict__ A,
                                          const half* __restrict__ x,
                                          half* __restrict__ out, int64_t N, int64_t K) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= N) return;

  float acc = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    acc = fmaf(__half2float(x[k]), __half2float(A[k * N + col]), acc);
  }
  out[col] = __float2half(acc);
}

}  // namespace

void gemv_fp16(const half* A, const half* x, half* out, std::int64_t N, std::int64_t K,
               cudaStream_t stream) {
  ENGINE_CHECK(A != nullptr, "gemv_fp16: A is null");
  ENGINE_CHECK(x != nullptr, "gemv_fp16: x is null");
  ENGINE_CHECK(out != nullptr, "gemv_fp16: out is null");
  ENGINE_CHECK(N >= 0, "gemv_fp16: N must be non-negative");
  ENGINE_CHECK(K >= 0, "gemv_fp16: K must be non-negative");
  if (N == 0 || K == 0) return;

  const bool is_aligned_64 = (N % 64 == 0) && (K % 8 == 0) &&
                             (reinterpret_cast<uintptr_t>(A) % sizeof(half2) == 0) &&
                             (reinterpret_cast<uintptr_t>(out) % sizeof(half2) == 0);

  if (is_aligned_64) {
    constexpr int COLS_PER_BLOCK = 64;
    constexpr int THREADS = 256;  // 8 warps
    const int blocks = static_cast<int>((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
    gemv_fp16_fast_k8_n64<<<blocks, THREADS, 0, stream>>>(A, x, out, N, K);
  } else {
    constexpr int THREADS = 256;
    const int blocks = static_cast<int>((N + THREADS - 1) / THREADS);
    gemv_fp16_scalar_fallback<<<blocks, THREADS, 0, stream>>>(A, x, out, N, K);
  }

  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
