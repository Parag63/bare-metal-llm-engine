//===----------------------------------------------------------------------===//
// kernels/gemv.cu -- Dense matrix-vector product for decode token generation.
//
// Computes out[N] = x[K] * A[K x N], row-major.
// Equivalent to matmul with M=1:
//   out[j] = sum_{k=0}^{K-1} x[k] * A[k * N + j]
//
// WHY GEMV MATTERS FOR LLM INFERENCE (DECODE PHASE):
//
// During autoregressive decode, the model generates one token at a time. The
// sequence length is M = 1. Launching matmul_tiled (which is tiled for 2D MxN
// matrices) at M=1 wastes ~97% of the thread block because only the first row
// of the 32x32 tile does useful work.
//
// GEMV parallelizes across the N output columns and K reduction dimension.
// For K=4096, N=4096:
//   - Ideal DRAM traffic: 64 MiB (matrix A) + 16 KiB (x) + 16 KiB (out) ≈ 64.03 MiB
//   - FLOPs: 2 * 4096 * 4096 = 33.55 MFLOP
//   - Arithmetic intensity: ~0.5 FLOP/byte (massively memory-bound)
//
// Goal: stream matrix A from DRAM at 85-90% of peak bandwidth (430-460 GB/s on
// RTX 4070 SUPER), executing in ~140 microseconds.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

// Fast path for N % 64 == 0 and K % 8 == 0:
// 256 threads per block = 8 warps.
// Each block computes 64 output columns.
// Each thread handles 2 columns using float2 vector memory instructions.
// 8 warps cooperatively reduce over the K dimension via shared memory.
__global__ void gemv_fast_k8_n64(const float* __restrict__ A, const float* __restrict__ x,
                                 float* __restrict__ out, int64_t N, int64_t K) {
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
    // Both columns valid and 64-bit aligned
    for (int64_t k = warp_id; k < K; k += WARPS) {
      const float xk = x[k];
      const float2 a_val = *reinterpret_cast<const float2*>(&A[k * N + c0]);
      acc0 += xk * a_val.x;
      acc1 += xk * a_val.y;
    }
  } else if (c0 < N) {
    for (int64_t k = warp_id; k < K; k += WARPS) {
      const float xk = x[k];
      acc0 += xk * A[k * N + c0];
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
      *reinterpret_cast<float2*>(&out[c0]) = make_float2(sum0, sum1);
    } else if (c0 < N) {
      out[c0] = sum0;
    }
  }
}

// Fallback kernel for arbitrary N and K
__global__ void gemv_scalar_fallback(const float* __restrict__ A,
                                     const float* __restrict__ x, float* __restrict__ out,
                                     int64_t N, int64_t K) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= N) return;

  double acc = 0.0;
  for (int64_t k = 0; k < K; ++k) {
    acc += static_cast<double>(x[k]) * static_cast<double>(A[k * N + col]);
  }
  out[col] = static_cast<float>(acc);
}

}  // namespace

void gemv(const float* A, const float* x, float* out, std::int64_t N, std::int64_t K,
          cudaStream_t stream) {
  ENGINE_CHECK(A != nullptr, "gemv: A is null");
  ENGINE_CHECK(x != nullptr, "gemv: x is null");
  ENGINE_CHECK(out != nullptr, "gemv: out is null");
  ENGINE_CHECK(N >= 0, "gemv: N must be non-negative");
  ENGINE_CHECK(K >= 0, "gemv: K must be non-negative");
  if (N == 0 || K == 0) return;

  // If aligned to 64 columns and K is a multiple of 8, use the multi-warp vectorized path
  const bool is_aligned_64 = (N % 64 == 0) && (K % 8 == 0) &&
                             (reinterpret_cast<uintptr_t>(A) % sizeof(float2) == 0) &&
                             (reinterpret_cast<uintptr_t>(out) % sizeof(float2) == 0);

  if (is_aligned_64) {
    constexpr int COLS_PER_BLOCK = 64;
    constexpr int THREADS = 256;  // 8 warps
    const int blocks = static_cast<int>((N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
    gemv_fast_k8_n64<<<blocks, THREADS, 0, stream>>>(A, x, out, N, K);
  } else {
    constexpr int THREADS = 256;
    const int blocks = static_cast<int>((N + THREADS - 1) / THREADS);
    gemv_scalar_fallback<<<blocks, THREADS, 0, stream>>>(A, x, out, N, K);
  }

  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
