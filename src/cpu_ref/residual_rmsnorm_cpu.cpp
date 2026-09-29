// CPU reference: Fused residual-add + RMSNorm (Exercise 8).
//
// Computes:
//   sum_out[r][c]  = x[r][c] + residual[r][c]
//   norm_out[r][c] = sum_out[r][c] / sqrt(mean(sum_out[r][:]^2) + eps) * weight[c]
//
// Both outputs are written. sum_out is preserved for subsequent residual connections,
// and norm_out feeds into the next sub-layer (attention or FFN).
//
// Like the other CPU oracles, loops are straightforward scalar passes with double
// precision accumulators to ensure it serves as a trustworthy oracle.

#include <engine/cpu_ref.hpp>

#include <cmath>

namespace engine::cpu {

void residual_rmsnorm(const float* x, const float* residual,
                      const float* weight, float* norm_out, float* sum_out,
                      std::int64_t rows, std::int64_t cols, float eps) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src_x = x + r * cols;
    const float* src_res = residual ? (residual + r * cols) : nullptr;
    float* dst_norm = norm_out + r * cols;
    float* dst_sum = sum_out ? (sum_out + r * cols) : nullptr;

    // Pass 1: Elementwise add and sum-of-squares accumulation in FP64
    double sumsq = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) {
      const float s = src_x[c] + (src_res != nullptr ? src_res[c] : 0.0f);
      if (dst_sum != nullptr) {
        dst_sum[c] = s;
      }
      sumsq += static_cast<double>(s) * static_cast<double>(s);
    }

    const double mean_sq = sumsq / static_cast<double>(cols);
    const float scale = static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));

    // Pass 2: Scale and apply optional per-channel weight
    for (std::int64_t c = 0; c < cols; ++c) {
      const float s = (dst_sum != nullptr) ? dst_sum[c] : (src_x[c] + (src_res != nullptr ? src_res[c] : 0.0f));
      const float w = (weight != nullptr) ? weight[c] : 1.0f;
      dst_norm[c] = s * scale * w;
    }
  }
}

}  // namespace engine::cpu
