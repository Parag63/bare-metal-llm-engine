//===----------------------------------------------------------------------===//
// kernels/matmul_register_tiled.cu -- Register-tiled 2D GEMM (Phase 4).
//
// Computes C[M x N] = A[M x K] * B[K x N], row-major.
//
// MICROARCHITECTURE & ARITHMETIC INTENSITY:
//
// In simple shared-memory tiling (matmul_tiled), each thread computes a single
// output element. For every 2 FLOPs (1 FMA), the thread reads 4 bytes from s_A
// and 4 bytes from s_B (8 bytes total). The shared-memory arithmetic intensity is:
//   AI_smem = 2 FLOP / 8 bytes = 0.25 FLOP/byte
// This severely bottlenecks performance at ~2.4 TFLOP/s due to shared-memory bank
// bandwidth saturation and load latency stalls.
//
// Register Tiling introduces a 2D thread-tile hierarchy:
//   - Block Tile: BM = 128, BN = 128, BK = 8
//   - Thread Tile: TM = 8, TN = 8 (each thread computes an 8x8 = 64 submatrix in registers)
//   - Threads per block: (128/8) x (128/8) = 16 x 16 = 256 threads
//
// For each k-step in BK, each thread loads:
//   - 8 floats from s_A into registers
//   - 8 floats from s_B into registers
// And executes an outer product of 8 x 8 = 64 FMAs (128 FLOPs) using 64 registers.
// Shared-memory arithmetic intensity jumps from 0.25 to:
//   AI_smem = 128 FLOP / (16 floats * 4 bytes) = 2.0 FLOP/byte (an 8x reduction in smem traffic!)
//
// Furthermore, global memory loads use 128-bit vector instructions (float4)
// to saturate the DRAM bus.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void matmul_reg_tiled_128x128_k8(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C, int64_t M, int64_t N,
                                            int64_t K) {
  // Transposed s_A[BK][BM] avoids bank conflicts during column-vector loads
  __shared__ float s_A[BK][BM];
  __shared__ float s_B[BK][BN];

  const int tid = threadIdx.y * 16 + threadIdx.x;
  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * BN;

  // Thread tile position within the block
  const int thread_m = threadIdx.y * TM;
  const int thread_n = threadIdx.x * TN;

  // Global load mapping for Matrix A: 256 threads load 128 x 8 = 1024 floats (4 floats / thread)
  const int load_a_row = tid / 2;        // 0 .. 127
  const int load_a_col = (tid % 2) * 4;  // 0 or 4

  // Global load mapping for Matrix B: 256 threads load 8 x 128 = 1024 floats (4 floats / thread)
  const int load_b_row = tid / 32;        // 0 .. 7
  const int load_b_col = (tid % 32) * 4;  // 0, 4, 8, ... 124

  // Register accumulator tile: 8x8 = 64 floats stored directly in registers
  float reg_accum[TM][TN] = {0.0f};

  // Outer loop over K tiles
  for (int64_t k_offset = 0; k_offset < K; k_offset += BK) {
    // 1. Vectorized 128-bit global load for A -> store transposed in s_A
    {
      const int64_t ga_row = block_m + load_a_row;
      const int64_t ga_col = k_offset + load_a_col;
      float4 a_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (ga_row < M && ga_col + 3 < K && (K % 4 == 0) &&
          (reinterpret_cast<uintptr_t>(A) % 16 == 0)) {
        a_val = *reinterpret_cast<const float4*>(&A[ga_row * K + ga_col]);
      } else if (ga_row < M) {
        if (ga_col + 0 < K) a_val.x = A[ga_row * K + ga_col + 0];
        if (ga_col + 1 < K) a_val.y = A[ga_row * K + ga_col + 1];
        if (ga_col + 2 < K) a_val.z = A[ga_row * K + ga_col + 2];
        if (ga_col + 3 < K) a_val.w = A[ga_row * K + ga_col + 3];
      }
      s_A[load_a_col + 0][load_a_row] = a_val.x;
      s_A[load_a_col + 1][load_a_row] = a_val.y;
      s_A[load_a_col + 2][load_a_row] = a_val.z;
      s_A[load_a_col + 3][load_a_row] = a_val.w;
    }

    // 2. Vectorized 128-bit global load for B -> store in s_B
    {
      const int64_t gb_row = k_offset + load_b_row;
      const int64_t gb_col = block_n + load_b_col;
      float4 b_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (gb_row < K && gb_col + 3 < N && (N % 4 == 0) &&
          (reinterpret_cast<uintptr_t>(B) % 16 == 0)) {
        b_val = *reinterpret_cast<const float4*>(&B[gb_row * N + gb_col]);
      } else if (gb_row < K) {
        if (gb_col + 0 < N) b_val.x = B[gb_row * N + gb_col + 0];
        if (gb_col + 1 < N) b_val.y = B[gb_row * N + gb_col + 1];
        if (gb_col + 2 < N) b_val.z = B[gb_row * N + gb_col + 2];
        if (gb_col + 3 < N) b_val.w = B[gb_row * N + gb_col + 3];
      }
      *reinterpret_cast<float4*>(&s_B[load_b_row][load_b_col]) = b_val;
    }

    __syncthreads();

// 3. Register tile outer products over BK
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float reg_a[TM];
      float reg_b[TN];

// Load 8 elements from s_A (bank-conflict free)
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        reg_a[m] = s_A[k][thread_m + m];
      }

// Load 8 elements from s_B
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        reg_b[n] = s_B[k][thread_n + n];
      }

// Outer product accumulation in registers
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          reg_accum[m][n] = fmaf(reg_a[m], reg_b[n], reg_accum[m][n]);
        }
      }
    }

    __syncthreads();
  }

// 4. Write back register accumulators to C
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int64_t row = block_m + thread_m + m;
    const int64_t col = block_n + thread_n;

    if (row < M) {
      if (col + 7 < N && (N % 4 == 0) && (reinterpret_cast<uintptr_t>(C) % 16 == 0)) {
        // Fast path: two 128-bit stores
        float4 c0 = make_float4(reg_accum[m][0], reg_accum[m][1], reg_accum[m][2],
                                reg_accum[m][3]);
        float4 c1 = make_float4(reg_accum[m][4], reg_accum[m][5], reg_accum[m][6],
                                reg_accum[m][7]);
        *reinterpret_cast<float4*>(&C[row * N + col + 0]) = c0;
        *reinterpret_cast<float4*>(&C[row * N + col + 4]) = c1;
      } else {
        // Boundary fallback
        for (int n = 0; n < TN; ++n) {
          if (col + n < N) {
            C[row * N + col + n] = reg_accum[m][n];
          }
        }
      }
    }
  }
}

}  // namespace

void matmul_register_tiled(const float* A, const float* B, float* C, std::int64_t M,
                           std::int64_t N, std::int64_t K, cudaStream_t stream) {
  ENGINE_CHECK(A != nullptr, "matmul_register_tiled: A is null");
  ENGINE_CHECK(B != nullptr, "matmul_register_tiled: B is null");
  ENGINE_CHECK(C != nullptr, "matmul_register_tiled: C is null");
  ENGINE_CHECK(M >= 0, "matmul_register_tiled: M must be non-negative");
  ENGINE_CHECK(N >= 0, "matmul_register_tiled: N must be non-negative");
  ENGINE_CHECK(K >= 0, "matmul_register_tiled: K must be non-negative");
  if (M == 0 || N == 0 || K == 0) return;

  dim3 block(16, 16);  // 256 threads
  dim3 grid(static_cast<unsigned int>((N + BN - 1) / BN),
            static_cast<unsigned int>((M + BM - 1) / BM));

  matmul_reg_tiled_128x128_k8<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
