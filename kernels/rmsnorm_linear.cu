//===----------------------------------------------------------------------===//
// kernels/rmsnorm_linear.cu -- EXERCISE 7. Fused RMSNorm + Linear Projection.
//
// Spec:      temp[r][k] = in[r][k] / sqrt(mean(in[r][:]^2) + eps) * rms_weight[k]
//            out[r][n]  = sum_k temp[r][k] * W[k][n]
// Oracle:    engine::cpu::rmsnorm_linear
// Test:      build/bin/engine_tests --filter=kernels.rmsnorm_linear
//
// Transformers execute RMSNorm followed immediately by linear projection at:
//   - pre-attention: hidden -> RMSNorm -> QKV projection
//   - pre-feedforward: hidden -> RMSNorm -> gate/up projection
// For LLaMA-7B, that is 64 kernel launches per token where 32 would suffice.
//
// Unfused execution requires:
//   1. Launch rmsnorm: read in[M x K], write temp[M x K] to DRAM.
//   2. Launch matmul: read temp[M x K], read W[K x N], write out[M x N] to DRAM.
// At M=512, K=4096: temp is 512 x 4096 x 4B = 8 MiB written to DRAM, then read right back.
//
// Fused execution computes the normalized row in shared memory and immediately
// accumulates the linear projection without ever writing temp to DRAM.
//
// Thread & Block Mapping:
//   Each block computes a 1 x kBlockSize strip of the output for row r:
//   1. Cooperative block-strided load of row r from `in` into shared memory.
//   2. Block reduction computes sum of squares; scale = rsqrtf(mean_sq + eps).
//   3. In-place normalization in shared memory: s_row[k] *= scale * rms_weight[k].
//   4. Matmul accumulation: each thread computes one output column col = col_start + tid.
//      For each k in 0..K-1: s_row[k] is broadcast to all threads in the warp,
//      while W[k * N + col] is loaded from global memory in coalesced 128-byte transactions.
//   5. Output written to out[r * N + col].
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cmath>

namespace engine::cuda {
namespace {

constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// WITHIN-BLOCK SUM REDUCTION: shared-memory tree + warp shuffle + broadcast.
//===----------------------------------------------------------------------===//
__device__ float block_reduce_sum(float* sdata, int tid) {
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  float val = sdata[tid];
  if (tid < 32) {
    val += sdata[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }

  if (tid == 0) {
    sdata[0] = val;
  }
  __syncthreads();
  val = sdata[0];
  __syncthreads();

  return val;
}

//===----------------------------------------------------------------------===//
// KERNEL: Fused RMSNorm + Linear projection.
//
// Dynamic shared memory layout:
//   smem[0 .. K-1]             : normalized row activations (s_row)
//   smem[K .. K+kBlockSize-1]  : scratch space for block reduction (s_reduce)
//===----------------------------------------------------------------------===//
__global__ void rmsnorm_linear_kernel(const float* __restrict__ in,
                                      const float* __restrict__ rms_weight,
                                      const float* __restrict__ W,
                                      float* __restrict__ out,
                                      std::int64_t M, std::int64_t N,
                                      std::int64_t K, float eps) {
  extern __shared__ float smem[];
  float* s_row = smem;
  float* s_reduce = smem + K;

  const int tid = threadIdx.x;
  const float inv_K = 1.0f / static_cast<float>(K);
  const std::int64_t col_start = static_cast<std::int64_t>(blockIdx.x) * kBlockSize;
  const std::int64_t col = col_start + tid;

  for (std::int64_t r = blockIdx.y; r < M; r += gridDim.y) {
    const float* row_in = in + r * K;
    float* row_out = out + r * N;

    // Pass 1: Cooperatively load row r into shared memory
    for (std::int64_t k = tid; k < K; k += blockDim.x) {
      s_row[k] = row_in[k];
    }
    __syncthreads();

    // Pass 2: Accumulate sum of squares and block reduce
    float thread_sum_sq = 0.0f;
    for (std::int64_t k = tid; k < K; k += blockDim.x) {
      const float val = s_row[k];
      thread_sum_sq += val * val;
    }
    s_reduce[tid] = thread_sum_sq;
    __syncthreads();

    const float row_sum_sq = block_reduce_sum(s_reduce, tid);
    const float mean_sq = row_sum_sq * inv_K;
    const float scale = rsqrtf(mean_sq + eps);

    // Pass 3: In-place normalize in shared memory
    if (rms_weight != nullptr) {
      for (std::int64_t k = tid; k < K; k += blockDim.x) {
        s_row[k] = s_row[k] * scale * rms_weight[k];
      }
    } else {
      for (std::int64_t k = tid; k < K; k += blockDim.x) {
        s_row[k] = s_row[k] * scale;
      }
    }
    __syncthreads();

    // Pass 4: Matmul accumulation for output column `col`
    if (col < N) {
      float acc = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        acc += s_row[k] * W[k * N + col];
      }
      row_out[col] = acc;
    }

    // Barrier: ensure all threads finish reading s_row before next row overwrites it
    __syncthreads();
  }
}

}  // namespace

void rmsnorm_linear(const float* in, const float* rms_weight,
                    const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K,
                    float eps, cudaStream_t stream) {
  ENGINE_CHECK(M >= 0 && N >= 0 && K >= 0, "rmsnorm_linear: negative dimension");
  ENGINE_CHECK(in != nullptr && W != nullptr && out != nullptr,
               "rmsnorm_linear: null device pointer");
  ENGINE_CHECK(eps >= 0.0f, "rmsnorm_linear: eps must be non-negative");
  if (M == 0 || N == 0) return;

  // An Mx0 times 0xN product is the MxN zero matrix.
  if (K == 0) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, static_cast<std::size_t>(M * N) * sizeof(float), stream));
    return;
  }

  const std::size_t smem_bytes =
      static_cast<std::size_t>(K + kBlockSize) * sizeof(float);

  if (smem_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(rmsnorm_linear_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 block(kBlockSize);
  const unsigned int grid_x = static_cast<unsigned int>((N + kBlockSize - 1) / kBlockSize);
  const unsigned int grid_y = static_cast<unsigned int>(M <= 65535 ? M : 65535);
  dim3 grid(grid_x, grid_y);

  rmsnorm_linear_kernel<<<grid, block, smem_bytes, stream>>>(
      in, rms_weight, W, out, M, N, K, eps);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
