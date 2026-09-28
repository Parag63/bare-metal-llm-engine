//===----------------------------------------------------------------------===//
// kernels/rmsnorm.cu -- EXERCISE 4. Your implementation.
//
// Spec:      out[r][c] = in[r][c] / sqrt(mean(in[r][:]^2) + eps) * weight[c]
// Oracle:    engine::cpu::rmsnorm
// Test:      build/bin/engine_tests --filter=kernels.rmsnorm
//              -> kernels.rmsnorm_matches_reference
//                 kernels.rmsnorm_of_zeros_is_zeros_not_nan
//
// Structurally this is easier than softmax -- one reduction instead of two, no
// numerical-stability subtlety -- so it should go quickly if exercises 2 and 3 left
// you with a reusable block-reduction helper. If it does not go quickly, that is a
// sign the reduction was not properly factored out, which is worth fixing now
// rather than at the fifth copy.
//
// It is also the normalisation LLaMA actually uses, so this kernel ships in the
// final engine and gets called twice per transformer layer (pre-attention and
// pre-feedforward) -- 64 times per token for a 32-layer model.
//===----------------------------------------------------------------------===//
//
// SUGGESTED DECOMPOSITION: one block per row, mirroring softmax.
//
//   1. Each thread accumulates a partial sum of squares over its strided share of
//      the row.
//   2. Block-reduce to the row total; compute scale = rsqrtf(total/cols + eps).
//   3. Each thread writes out[c] = in[c] * scale * weight[c].
//
// rsqrtf() is a single hardware instruction that computes 1/sqrt(x) directly, and is
// both faster and no less accurate here than 1.0f/sqrtf(x). Use it.
//
// PLACE eps INSIDE THE SQRT, matching the oracle:
//     scale = rsqrtf(mean_sq + eps)      not     rsqrtf(mean_sq) + eps
// Both appear in the wild. They differ only for near-zero rows, which is exactly
// what the test feeds you.
//
// A REAL SUBTLETY WORTH KNOWING
//
// The oracle accumulates the sum of squares in double. Your kernel will accumulate
// in float. For a 4096-wide row of activations that is normally fine, but squaring
// amplifies dynamic range -- a value of 1e20 squares to 1e40, which overflows FP32
// (max ~3.4e38) to +inf even though the input was representable. Production
// implementations sometimes accumulate in double or pre-scale the row. Not required
// here, but note it in the report: it is precisely the class of numerical issue that
// makes FP16 inference harder than FP32, and it shows you understood the trade-off
// rather than just matching a reference.
//
// PITFALLS
//   * weight == nullptr must mean "no scaling" (treat as 1.0f), not a crash. The test
//     checks both paths.
//   * cols > blockDim needs a strided loop, same as softmax.
//   * Dividing by cols as an int -- integer division truncates. Cast to float first.
//
// ACCEPTANCE
//   * Matches the oracle within rtol 1e-5 for shapes (1,4096), (128,4096), (32,127),
//     both with and without weight. 4096 is LLaMA-7B's hidden dimension, so that
//     shape is the one that matters.
//   * Handles an all-zeros row without producing NaN -- eps exists for that case.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cmath>

namespace engine::cuda {
namespace {

// 256 threads per block, matching vector_add, reduce_sum, and softmax.
constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// WITHIN-BLOCK SUM REDUCTION: shared-memory tree + warp shuffle + broadcast.
//
// Reuses the identical reduction pattern established in reduce_sum and softmax:
//   1. Tree reduction across warps in shared memory (halving active threads).
//   2. Single-warp reduction using register shuffle down (__shfl_down_sync).
//   3. Thread 0 writes the scalar result to sdata[0] and broadcasts to all
//      threads in the block via a shared-memory barrier.
//
// Precondition: sdata[0..kBlockSize) is populated by the block's threads.
// Postcondition: all threads in the block return the reduced scalar;
//                sdata is clean and safe to reuse.
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

  // Broadcast lane 0's result to all threads in the block via sdata[0].
  if (tid == 0) {
    sdata[0] = val;
  }
  __syncthreads();
  val = sdata[0];
  __syncthreads();

  return val;
}

//===----------------------------------------------------------------------===//
// KERNEL: Row-wise Root Mean Square Normalization (RMSNorm).
//
// Decomposition: Each block independently handles rows in a grid-stride loop.
// Within each row:
//   Pass 1: Accumulate sum of squares across columns via thread-local stride.
//   Pass 2: Block-reduce to row total; compute scale = rsqrtf(mean_sq + eps).
//   Pass 3: Scale each element by scale and multiply by weight[c] (if provided).
//
// Note: eps is placed INSIDE the sqrt, matching the reference:
//   scale = rsqrtf(mean_sq + eps)
// This ensures that all-zero rows produce finite scale and 0.0f output rather
// than 0/0 = NaN.
//===----------------------------------------------------------------------===//
__global__ void rmsnorm_kernel(const float* __restrict__ in,
                               const float* __restrict__ weight,
                               float* __restrict__ out,
                               std::int64_t rows, std::int64_t cols, float eps) {
  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;
  const float inv_cols = 1.0f / static_cast<float>(cols);

  for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const float* row_in = in + r * cols;
    float* row_out = out + r * cols;

    // Pass 1: Accumulate sum of squares in thread-local register.
    float thread_sum_sq = 0.0f;
    for (std::int64_t c = tid; c < cols; c += blockDim.x) {
      const float x = row_in[c];
      thread_sum_sq += x * x;
    }
    sdata[tid] = thread_sum_sq;
    __syncthreads();

    // Pass 2: Block reduction to find row total sum of squares.
    const float row_sum_sq = block_reduce_sum(sdata, tid);

    // Compute normalization scale: 1.0 / sqrt(mean_sq + eps).
    // Cast cols to float to avoid integer truncation. rsqrtf is a single hardware op.
    const float mean_sq = row_sum_sq * inv_cols;
    const float scale = rsqrtf(mean_sq + eps);

    // Pass 3: Normalize and apply optional learned weight vector.
    if (weight != nullptr) {
      for (std::int64_t c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_in[c] * scale * weight[c];
      }
    } else {
      for (std::int64_t c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_in[c] * scale;
      }
    }

    // Synchronize so no thread overwrites sdata for the next row prematurely.
    __syncthreads();
  }
}

}  // namespace

void rmsnorm(const float* in, const float* weight, float* out, std::int64_t rows,
             std::int64_t cols, float eps, cudaStream_t stream) {
  ENGINE_CHECK(rows >= 0 && cols >= 0, "rmsnorm: negative dimension");
  ENGINE_CHECK(in != nullptr && out != nullptr, "rmsnorm: null device pointer");
  ENGINE_CHECK(eps >= 0.0f, "rmsnorm: eps must be non-negative");
  if (rows == 0 || cols == 0) return;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // Size grid to keep SMs busy without launching excessive blocks.
  // One block per row when rows <= num_sms * 32; grid-stride handles larger rows.
  const std::int64_t blocks_needed = rows;
  const std::int64_t blocks_wanted = static_cast<std::int64_t>(num_sms) * 32;
  const int grid = static_cast<int>(blocks_needed < blocks_wanted ? blocks_needed
                                                                  : blocks_wanted);

  rmsnorm_kernel<<<grid, kBlockSize, 0, stream>>>(in, weight, out, rows, cols, eps);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
