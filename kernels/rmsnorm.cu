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

namespace engine::cuda {
namespace {

// TODO(parag): your kernel here.
//
// __global__ void rmsnorm_kernel(const float* __restrict__ in,
//                                const float* __restrict__ weight,
//                                float* __restrict__ out,
//                                std::int64_t rows, std::int64_t cols, float eps) {...}

}  // namespace

void rmsnorm(const float* in, const float* weight, float* out, std::int64_t rows,
             std::int64_t cols, float eps, cudaStream_t stream) {
  ENGINE_CHECK(rows >= 0 && cols >= 0, "rmsnorm: negative dimension");
  ENGINE_CHECK(in != nullptr && out != nullptr, "rmsnorm: null device pointer");
  ENGINE_CHECK(eps >= 0.0f, "rmsnorm: eps must be non-negative");
  if (rows == 0 || cols == 0) return;

  // TODO(parag): remove this line and implement. `weight` may legitimately be null.
  (void)stream;
  ENGINE_CHECK(false, "rmsnorm: not implemented yet (exercise 4)");
}

}  // namespace engine::cuda
