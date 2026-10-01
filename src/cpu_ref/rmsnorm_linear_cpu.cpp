// CPU reference: Fused RMSNorm + Linear projection (Exercise 7).
//
// Computes:
//   temp[r][k] = in[r][k] / sqrt(mean(in[r][:]^2) + eps) * rms_weight[k]
//   out[r][n]  = sum_k temp[r][k] * W[k][n]
//
// Eliminates the intermediate M x K activation buffer from global memory.
// In this CPU oracle, row-by-row execution is used with FP64 accumulators for
// the dot products and RMSNorm reduction.

#include <engine/cpu_ref.hpp>

#include <cmath>
#include <vector>

namespace engine::cpu {

void rmsnorm_linear(const float* in, const float* rms_weight, const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K, float eps) {
  std::vector<float> temp(static_cast<std::size_t>(K));

  for (std::int64_t r = 0; r < M; ++r) {
    const float* src = in + r * K;
    float* dst = out + r * N;

    // 1. RMSNorm for row r into temp buffer
    double sumsq = 0.0;
    for (std::int64_t k = 0; k < K; ++k) {
      sumsq += static_cast<double>(src[k]) * static_cast<double>(src[k]);
    }

    const double mean_sq = sumsq / static_cast<double>(K);
    const float scale =
        static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));

    for (std::int64_t k = 0; k < K; ++k) {
      const float w = (rms_weight != nullptr) ? rms_weight[k] : 1.0f;
      temp[static_cast<std::size_t>(k)] = src[k] * scale * w;
    }

    // 2. Matrix multiplication: dst[n] = sum_k temp[k] * W[k * N + n]
    for (std::int64_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::int64_t k = 0; k < K; ++k) {
        acc += static_cast<double>(temp[static_cast<std::size_t>(k)]) *
               static_cast<double>(W[k * N + n]);
      }
      dst[n] = static_cast<float>(acc);
    }
  }
}

}  // namespace engine::cpu
