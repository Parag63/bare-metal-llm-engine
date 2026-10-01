//===----------------------------------------------------------------------===//
// src/cpu_ref/gemv_cpu.cpp -- CPU reference for matrix-vector product.
//
// Computes out[N] = x[K] * A[K x N], all row-major.
// Equivalent to matmul with M=1:
//   out[j] = sum_{k=0}^{K-1} x[k] * A[k * N + j]
//
// Scalar loops with FP64 accumulators for trustworthy reference values.
//===----------------------------------------------------------------------===//

#include <engine/cpu_ref.hpp>

namespace engine::cpu {

void gemv(const float* A, const float* x, float* out, std::int64_t N, std::int64_t K) {
  for (std::int64_t j = 0; j < N; ++j) {
    double acc = 0.0;
    for (std::int64_t k = 0; k < K; ++k) {
      acc += static_cast<double>(x[k]) * static_cast<double>(A[k * N + j]);
    }
    out[j] = static_cast<float>(acc);
  }
}

void gemv_fp16(const half* A, const half* x, half* out, std::int64_t N, std::int64_t K) {
  for (std::int64_t j = 0; j < N; ++j) {
    double acc = 0.0;
    for (std::int64_t k = 0; k < K; ++k) {
      double xk = static_cast<double>(half_to_float(x[k]));
      double Akj = static_cast<double>(half_to_float(A[k * N + j]));
      acc += xk * Akj;
    }
    out[j] = float_to_half(static_cast<float>(acc));
  }
}

}  // namespace engine::cpu
