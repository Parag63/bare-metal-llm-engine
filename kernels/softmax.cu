//===----------------------------------------------------------------------===//
// kernels/softmax.cu -- EXERCISE 3. Your implementation.
//
// Spec:      row-wise softmax of a rows x cols row-major matrix
// Oracle:    engine::cpu::softmax_rows  (read it first -- it explains the maths)
// Test:      build/bin/engine_tests --filter=kernels.softmax
//              -> kernels.softmax_rows_matches_reference
//                 kernels.softmax_rows_survives_the_edge_cases
//
// This is the direct ancestor of FlashAttention. Attention is
// softmax(QK^T / sqrt(d)) V, and the reason attention costs O(n^2) MEMORY rather
// than merely O(n^2) compute is that a naive implementation materialises the whole
// score matrix so it can take a softmax over each row. Everything you learn here
// about restructuring that softmax carries straight into January.
//===----------------------------------------------------------------------===//
//
// SUGGESTED DECOMPOSITION: ONE BLOCK PER ROW.
//
// Each block handles one row independently, so no cross-block communication is
// needed and the whole thing is one launch: grid = rows, block = 256 threads. Within
// a block you need two reductions (max, then sum), which is precisely the machinery
// from exercise 2 -- factor it into a reusable __device__ helper rather than writing
// it a third time in exercise 4.
//
// STEP 1 -- three-pass version. Get this correct first.
//   1. Block-reduce to find the row max.
//   2. Each thread computes exp(x - max) for its elements, accumulating a partial
//      sum; block-reduce those to the row total.
//   3. Each thread divides its elements by the total.
//
//   Note this reads the row from global memory three times (or twice, if you cache
//   values in registers between steps 2 and 3).
//
// STEP 2 -- the online (single-pass) version. Attempt after step 1 passes.
//
//   Maintain a running max m and running sum s, and update both as each new value
//   arrives. The identity that makes this work: when a value larger than the current
//   max appears, every term already accumulated in s was scaled by the OLD max and
//   must be corrected:
//
//       m_new = max(m_old, x)
//       s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
//
//   Check the two cases by hand. If x <= m_old then m_new == m_old, the exp factor
//   is 1, and this reduces to a plain accumulation. If x > m_old then m_new == x, the
//   factor exp(m_old - x) < 1 rescales the old sum into the new frame, and the new
//   term is exp(0) == 1.
//
//   This same pair of running values, combined tile-by-tile instead of element-by-
//   element, IS FlashAttention. Implementing it here at row scale means January's
//   work is a generalisation rather than a new idea.
//
//===----------------------------------------------------------------------===//
// PITFALLS
//
//   * Initialising the running max to 0.0f. Use -INFINITY. Rows of entirely negative
//     values otherwise come out wrong, and your test data may not contain such a row
//     unless you deliberately add one. (The test does.)
//   * Using expf() on unshifted inputs. exp(x) overflows to +inf for x > ~88.7 in
//     FP32, then inf/inf = NaN. Always subtract the max first.
//   * Divergence in the reduction when cols is not a multiple of blockDim.
//   * cols > blockDim: each thread must handle several columns via a strided loop.
//     Do not assume one thread per column.
//   * Rows that are entirely -INFINITY (fully masked, which happens in real
//     attention) produce 0/0. Decide the behaviour and comment it.
//
// ACCEPTANCE
//   * Matches the oracle within rtol 1e-5 for shapes (1,1), (1,1024), (128,127),
//     (7,4096), and (32,50257) -- that last one is a realistic vocabulary size, which
//     is the shape the final output projection actually produces.
//   * Every output row sums to 1.0 within 1e-5.
//   * Correct on a row of all-negative values and on a row containing +300.0f.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cmath>

namespace engine::cuda {
namespace {

// 256 threads per block, matching vector_add and reduce_sum.
// A multiple of the 32-wide warp, small enough for good occupancy,
// large enough to amortise per-block costs.
constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// WITHIN-BLOCK REDUCTIONS: shared-memory tree + warp shuffle + broadcast.
//
// Both helpers follow the reduction pattern from kernels/reduce_sum.cu:
//   1. Tree reduction across warps in shared memory (halving active threads).
//   2. Single-warp reduction using register shuffle down (__shfl_down_sync).
//   3. Thread 0 writes the scalar result to sdata[0] and broadcasts to all
//      threads in the block via a shared-memory barrier.
//
// Precondition: sdata[0..kBlockSize) is populated by the block's threads.
// Postcondition: all threads in the block return the reduced scalar;
//                sdata is clean and safe to reuse.
//===----------------------------------------------------------------------===//

__device__ float block_reduce_max(float* sdata, int tid) {
  // Tree reduction in shared memory for upper warps.
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  // Single-warp reduction using shuffle down.
  float val = sdata[tid];
  if (tid < 32) {
    val = fmaxf(val, sdata[tid + 32]);
    for (int offset = 16; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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

__device__ float block_reduce_sum(float* sdata, int tid) {
  // Tree reduction in shared memory for upper warps.
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Single-warp reduction using shuffle down.
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
// KERNEL: Numerically stable row-wise softmax (Three-pass block implementation).
//
// Decomposition: Each block independently handles rows in a grid-stride loop.
// Within each row:
//   Pass 1: Find row maximum across columns (thread-local stride + block_reduce_max).
//   Pass 2: Compute exp(x - max) and accumulate sum (thread-local stride + block_reduce_sum).
//   Pass 3: Multiply each exp(x - max) by (1.0 / sum) and write to out.
//
// Numerical stability:
//   Subtracting the row maximum ensures that the argument to expf() is <= 0.0f.
//   In FP32, exp(x) overflows to +inf for x > ~88.7f, which would cause inf/inf = NaN.
//   With the max subtraction, intermediate values are in (0.0, 1.0], and because the
//   maximum element has x - max == 0.0f, its contribution is exp(0) == 1.0f, ensuring
//   the sum is >= 1.0f and preventing division by zero.
//===----------------------------------------------------------------------===//
__global__ void softmax_rows_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    std::int64_t rows, std::int64_t cols) {
  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const float* row_in = in + r * cols;
    float* row_out = out + r * cols;

    // Pass 1: Row maximum across columns.
    // Initialize to -INFINITY so all-negative rows work correctly.
    float thread_max = -INFINITY;
    for (std::int64_t c = tid; c < cols; c += blockDim.x) {
      thread_max = fmaxf(thread_max, row_in[c]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    const float row_max = block_reduce_max(sdata, tid);

    // Guard against rows of entirely -INFINITY (e.g., completely masked attention rows).
    // If all elements are -inf, exp(-inf - (-inf)) is NaN. We output zeros everywhere.
    if (row_max == -INFINITY) {
      for (std::int64_t c = tid; c < cols; c += blockDim.x) {
        row_out[c] = 0.0f;
      }
      __syncthreads();
      continue;
    }

    // Pass 2: Exponentiate shifted elements and accumulate their sum.
    // Use expf() (sub-ulp accuracy) to satisfy kSoftmaxRtol = 1e-5.
    float thread_sum = 0.0f;
    for (std::int64_t c = tid; c < cols; c += blockDim.x) {
      thread_sum += expf(row_in[c] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    const float row_sum = block_reduce_sum(sdata, tid);

    // Pass 3: Normalize by dividing by the row total.
    const float inv_sum = 1.0f / row_sum;
    for (std::int64_t c = tid; c < cols; c += blockDim.x) {
      row_out[c] = expf(row_in[c] - row_max) * inv_sum;
    }

    // Ensure all threads finish writing row_out before any thread modifies sdata
    // in the next row iteration.
    __syncthreads();
  }
}

}  // namespace

void softmax_rows(const float* in, float* out, std::int64_t rows, std::int64_t cols,
                  cudaStream_t stream) {
  ENGINE_CHECK(rows >= 0 && cols >= 0, "softmax_rows: negative dimension");
  ENGINE_CHECK(in != nullptr && out != nullptr, "softmax_rows: null device pointer");
  if (rows == 0 || cols == 0) return;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // Size grid to keep SMs busy without launching excessive blocks.
  // One block per row when rows <= num_sms * 32; grid-stride handles larger rows.
  const std::int64_t blocks_needed = rows;
  const std::int64_t blocks_wanted = static_cast<std::int64_t>(num_sms) * 32;
  const int grid =
      static_cast<int>(blocks_needed < blocks_wanted ? blocks_needed : blocks_wanted);

  softmax_rows_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, rows, cols);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
