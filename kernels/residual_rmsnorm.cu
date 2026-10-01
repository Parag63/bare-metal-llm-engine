//===----------------------------------------------------------------------===//
// kernels/residual_rmsnorm.cu -- EXERCISE 8. Fused Residual-Add + RMSNorm.
//
// Spec:      sum_out[r][c]  = x[r][c] + residual[r][c]
//            norm_out[r][c] = sum_out[r][c] / sqrt(mean(sum_out[r][:]^2) + eps) * weight[c]
// Oracle:    engine::cpu::residual_rmsnorm
// Test:      build/bin/engine_tests --filter=kernels.residual_rmsnorm
//
// Transformers execute residual-add followed immediately by normalisation at
// every sub-layer boundary (post-attention and post-feedforward) -- 64 times
// per token in a 32-layer LLaMA model.
//
// Unfused execution requires:
//   1. Launch vector_add: read x, read residual, write sum to DRAM.
//   2. Launch rmsnorm: read sum from DRAM, write norm_out to DRAM.
// Total DRAM traffic: 3 reads + 2 writes = 5 * rows * cols * 4 bytes.
//
// Fused execution computes the residual sum and accumulates sum-of-squares in
// registers/shared memory in a single pass. Both sum_out (needed by the next
// residual connection) and norm_out (feeding the next sub-layer) are written in
// one kernel launch.
// Total DRAM traffic: 2 reads + 2 writes = 4 * rows * cols * 4 bytes.
// Eliminates 1 full read pass over the hidden dimension and 1 kernel launch.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cmath>

namespace engine::cuda {
namespace {

// 256 threads per block, matching vector_add, reduce_sum, softmax, and rmsnorm.
constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// WITHIN-BLOCK SUM REDUCTION: shared-memory tree + warp shuffle + broadcast.
// Identical to rmsnorm.cu.
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
// KERNEL: Fused Residual-Add + RMSNorm.
//
// Decomposition:
//   Pass 1: Each thread adds x[c] + residual[c], writes the result to sum_out[c],
//           and accumulates thread-local sum of squares.
//   Pass 2: Block reduction computes total sum of squares; scale = rsqrtf(mean_sq + eps).
//   Pass 3: Normalize and multiply by weight[c] (if provided), writing to norm_out[c].
//===----------------------------------------------------------------------===//
__global__ void residual_rmsnorm_kernel(const float* __restrict__ x,
                                        const float* __restrict__ residual,
                                        const float* __restrict__ weight,
                                        float* __restrict__ norm_out,
                                        float* __restrict__ sum_out, std::int64_t rows,
                                        std::int64_t cols, float eps) {
  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;
  const float inv_cols = 1.0f / static_cast<float>(cols);

  for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const float* row_x = x + r * cols;
    const float* row_res = residual + r * cols;
    float* row_norm = norm_out + r * cols;
    float* row_sum = sum_out + r * cols;

    // Pass 1: Elementwise add, write to sum_out, accumulate sum-of-squares in registers
    float thread_sum_sq = 0.0f;
    for (std::int64_t c = tid; c < cols; c += blockDim.x) {
      const float s = row_x[c] + row_res[c];
      row_sum[c] = s;
      thread_sum_sq += s * s;
    }
    sdata[tid] = thread_sum_sq;
    __syncthreads();

    // Pass 2: Block reduction to find row sum of squares
    const float row_sum_sq = block_reduce_sum(sdata, tid);
    const float mean_sq = row_sum_sq * inv_cols;
    const float scale = rsqrtf(mean_sq + eps);

    // Pass 3: Normalize and scale
    if (weight != nullptr) {
      for (std::int64_t c = tid; c < cols; c += blockDim.x) {
        row_norm[c] = row_sum[c] * scale * weight[c];
      }
    } else {
      for (std::int64_t c = tid; c < cols; c += blockDim.x) {
        row_norm[c] = row_sum[c] * scale;
      }
    }

    __syncthreads();
  }
}

}  // namespace

void residual_rmsnorm(const float* x, const float* residual, const float* weight,
                      float* norm_out, float* sum_out, std::int64_t rows,
                      std::int64_t cols, float eps, cudaStream_t stream) {
  ENGINE_CHECK(rows >= 0 && cols >= 0, "residual_rmsnorm: negative dimension");
  ENGINE_CHECK(
      x != nullptr && residual != nullptr && norm_out != nullptr && sum_out != nullptr,
      "residual_rmsnorm: null device pointer");
  ENGINE_CHECK(eps >= 0.0f, "residual_rmsnorm: eps must be non-negative");
  if (rows == 0 || cols == 0) return;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // Size grid to keep SMs busy without launching excessive blocks.
  const std::int64_t blocks_needed = rows;
  const std::int64_t blocks_wanted = static_cast<std::int64_t>(num_sms) * 32;
  const int grid =
      static_cast<int>(blocks_needed < blocks_wanted ? blocks_needed : blocks_wanted);

  residual_rmsnorm_kernel<<<grid, kBlockSize, 0, stream>>>(x, residual, weight, norm_out,
                                                           sum_out, rows, cols, eps);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
