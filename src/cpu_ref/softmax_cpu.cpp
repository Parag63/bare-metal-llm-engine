// CPU reference: numerically stable row-wise softmax.
//
// Three passes per row: max, then sum of exp(x - max), then normalise.
//
// Read this carefully, because exercise 3 asks you to fuse it and because
// FlashAttention (Objective 3, Jan 2027) is essentially the observation that these
// three passes can be collapsed into ONE by updating the running max and the
// running sum together:
//
//     when a new max m' > m arrives, the old sum must be rescaled:
//         sum' = sum * exp(m - m') + exp(x_new - m')
//
// That single rescaling identity is the whole trick behind online softmax, and
// FlashAttention is that trick applied tile-by-tile so the S = QK^T matrix never
// has to exist in global memory. Everything else in FlashAttention is bookkeeping.

#include <engine/cpu_ref.hpp>

#include <cmath>

namespace engine::cpu {

void softmax_rows(const float* in, float* out, std::int64_t rows, std::int64_t cols) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * cols;
    float* dst = out + r * cols;

    // Pass 1: row maximum. Without this, exp() overflows to +inf for any input
    // above ~88.7 in FP32, and the result becomes inf/inf = NaN.
    float m = src[0];
    for (std::int64_t c = 1; c < cols; ++c) {
      if (src[c] > m) m = src[c];
    }

    // Pass 2: exponentiate the shifted values and accumulate their sum.
    // Shifting by the max is exact in the sense that it cannot change the result:
    // exp(x_i - m) / sum_j exp(x_j - m) == exp(x_i) / sum_j exp(x_j), because the
    // factor exp(-m) cancels. It only changes the intermediate magnitudes.
    double sum = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) {
      const float e = std::exp(src[c] - m);
      dst[c] = e;
      sum += static_cast<double>(e);
    }

    // Pass 3: normalise. sum >= 1 always, because the max element contributes
    // exp(0) == 1, so there is no division-by-zero case to guard.
    const float inv = static_cast<float>(1.0 / sum);
    for (std::int64_t c = 0; c < cols; ++c) {
      dst[c] *= inv;
    }
  }
}

}  // namespace engine::cpu
