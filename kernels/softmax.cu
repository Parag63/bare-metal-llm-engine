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

namespace engine::cuda {
namespace {

// TODO(parag): your kernel here.
//
// __global__ void softmax_rows_kernel(const float* __restrict__ in,
//                                     float* __restrict__ out,
//                                     std::int64_t rows, std::int64_t cols) { ... }

}  // namespace

void softmax_rows(const float* in, float* out, std::int64_t rows, std::int64_t cols,
                  cudaStream_t stream) {
  ENGINE_CHECK(rows >= 0 && cols >= 0, "softmax_rows: negative dimension");
  ENGINE_CHECK(in != nullptr && out != nullptr, "softmax_rows: null device pointer");
  if (rows == 0 || cols == 0) return;

  // TODO(parag): remove this line and implement.
  (void)stream;
  ENGINE_CHECK(false, "softmax_rows: not implemented yet (exercise 3)");
}

}  // namespace engine::cuda
