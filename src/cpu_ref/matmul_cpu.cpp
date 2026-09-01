// CPU reference: dense matrix multiply, C[MxN] = A[MxK] * B[KxN], all row-major.
//
// The triple loop is intentionally the textbook i-j-k order. It has poor cache
// behaviour (the k-loop strides through B by N floats at a time), which makes it
// slow -- and that is fine. It is the oracle, not a competitor. Optimising it
// would risk introducing a bug into the one piece of code that must be trusted.
//
// Keep the plain O(M*N*K) FLOP count in mind for the benchmarks: 2*M*N*K
// floating-point operations (one multiply + one add per inner iteration). At
// M=N=K=1024 that is 2.1 GFLOP, which is the numerator when you report GFLOP/s.

#include <engine/cpu_ref.hpp>

namespace engine::cpu {

void matmul(const float* A, const float* B, float* C, std::int64_t M, std::int64_t N,
            std::int64_t K) {
  for (std::int64_t i = 0; i < M; ++i) {
    for (std::int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (std::int64_t k = 0; k < K; ++k) {
        // A is M x K -> element (i,k) at i*K + k
        // B is K x N -> element (k,j) at k*N + j
        acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
      }
      C[i * N + j] = static_cast<float>(acc);
    }
  }
}

}  // namespace engine::cpu
