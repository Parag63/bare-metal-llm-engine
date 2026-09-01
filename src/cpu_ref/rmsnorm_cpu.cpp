// CPU reference: RMSNorm (Zhang & Sennrich, 2019), the normalisation used by LLaMA.
//
//     out[r][c] = in[r][c] / sqrt( mean_c(in[r][c]^2) + eps ) * weight[c]
//
// Compare LayerNorm:
//     out = (in - mean) / sqrt(var + eps) * weight + bias
//
// RMSNorm drops the mean subtraction and the bias. That means one reduction per
// row instead of two, and no bias tensor to load -- cheaper in both compute and
// memory traffic, with no measured quality loss for transformer LMs. That
// cost/benefit argument is exactly the kind of thing to write up in the report.

#include <engine/cpu_ref.hpp>

#include <cmath>

namespace engine::cpu {

void rmsnorm(const float* in, const float* weight, float* out, std::int64_t rows,
             std::int64_t cols, float eps) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * cols;
    float* dst = out + r * cols;

    // Mean of squares, accumulated in double (same reasoning as reduce_sum).
    double sumsq = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) {
      sumsq += static_cast<double>(src[c]) * static_cast<double>(src[c]);
    }
    const double mean_sq = sumsq / static_cast<double>(cols);

    // Note eps is INSIDE the sqrt. Some implementations put it outside; that
    // changes results for near-zero rows. Match this convention in the CUDA
    // kernel or the tests will disagree with you at eps-scale tolerances.
    const float scale = static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));

    for (std::int64_t c = 0; c < cols; ++c) {
      const float w = (weight != nullptr) ? weight[c] : 1.0f;
      dst[c] = src[c] * scale * w;
    }
  }
}

}  // namespace engine::cpu
