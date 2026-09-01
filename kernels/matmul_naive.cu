//===----------------------------------------------------------------------===//
// kernels/matmul_naive.cu -- EXERCISE 5. Your implementation.
//
// Spec:      C[MxN] = A[MxK] * B[KxN], all row-major
// Oracle:    engine::cpu::matmul
// Test:      build/bin/engine_tests --filter=kernels.matmul
//              -> kernels.matmul_naive_matches_reference
//                 kernels.matmul_with_k_zero_is_the_zero_matrix
// Bench:     bench/bench_kernels -- keep this kernel forever as the baseline that
//            exercise 6 must beat. "3x faster than naive" is only meaningful if the
//            naive number is measured on the same hardware, same day.
//===----------------------------------------------------------------------===//
//
// THE SIMPLEST DECOMPOSITION: one thread per output element.
//
// Thread (row, col) computes the dot product of row `row` of A with column `col` of
// B. No cooperation, no shared memory -- structurally as simple as vector_add, just
// with a 2-D index and an inner loop.
//
//     dim3 block(16, 16);
//     dim3 grid((N + 15) / 16, (M + 15) / 16);
//
// Note the grid is sized by N in x and M in y. Mixing those up is the single most
// common bug here, and it only shows up when M != N -- which is why the test
// includes non-square shapes.
//
//===----------------------------------------------------------------------===//
// AN EXPERIMENT WORTH RUNNING, AND WRITING UP
//
// There are two ways to map threads to output elements:
//
//     (a)  row = blockIdx.y * blockDim.y + threadIdx.y;
//          col = blockIdx.x * blockDim.x + threadIdx.x;
//
//     (b)  row = blockIdx.x * blockDim.x + threadIdx.x;
//          col = blockIdx.y * blockDim.y + threadIdx.y;
//
// They compute exactly the same answer. One is several times faster.
//
// The reason is coalescing. threadIdx.x is the fastest-varying dimension, so lanes
// within a warp differ in x. Under (a) consecutive lanes have consecutive `col`, so
// their reads of B[k*N + col] and writes to C[row*N + col] are consecutive addresses
// -- one wide transaction per warp. Under (b) consecutive lanes have consecutive
// `row`, so those same accesses are N floats apart -- up to 32 separate transactions,
// most of each fetched cache line discarded.
//
// Implement (a). Then try (b), measure both, and record the ratio in your lab
// notebook. It is the cheapest possible demonstration that memory access patterns
// dominate GPU performance, and it makes a very good answer to "what did you learn
// about GPU programming" in an interview.
//
//===----------------------------------------------------------------------===//
// WHY THIS KERNEL IS SLOW -- the arithmetic that motivates exercise 6
//
// Each output element requires K elements of A and K of B: 2K floats = 8K bytes read,
// for 2K floating-point operations (K multiplies + K adds). Arithmetic intensity:
//
//     2K FLOP / 8K bytes  =  0.25 FLOP/byte
//
// The RTX 4090 delivers roughly 82 TFLOP/s of FP32 against 1008 GB/s, a balance point
// near 82 FLOP/byte. At 0.25 you are using well under 1% of the arithmetic units; the
// kernel spends essentially all its time waiting for memory.
//
// The waste is redundancy: every thread in a row-block re-reads the same row of A,
// and every thread in a column-block re-reads the same column of B. With M=N=K=1024,
// each element of A is read 1024 times. The L1 and L2 caches recover some of this by
// accident, which is why the naive kernel is not as catastrophic as the raw figure
// suggests -- but relying on cache luck is not a strategy. Exercise 6 makes the reuse
// explicit and deliberate.
//
// PITFALLS
//   * Bounds-check BOTH row < M and col < N. A partially covered tile is the norm.
//   * Accumulate in float, but be aware the oracle uses double. For K=1024 the
//     divergence is well inside rtol 1e-4; for K in the tens of thousands it would
//     not be. Comment on it rather than fighting it.
//   * A[i*K + k] and B[k*N + j] -- write the indices out on paper once. Confusing K
//     and N in the strides produces a kernel that is correct only for square inputs.
//
// ACCEPTANCE
//   * Matches the oracle within rtol 1e-4 for (M,N,K) = (1,1,1), (32,32,32),
//     (128,64,256), (17,23,31) and (512,512,512). The prime-ish shape catches
//     tiling and bounds bugs that every power-of-two shape hides.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {
namespace {

// TODO(parag): your kernel here.
//
// __global__ void matmul_naive_kernel(const float* __restrict__ A,
//                                     const float* __restrict__ B,
//                                     float* __restrict__ C,
//                                     std::int64_t M, std::int64_t N,
//                                     std::int64_t K) { ... }

}  // namespace

void matmul_naive(const float* A, const float* B, float* C, std::int64_t M,
                  std::int64_t N, std::int64_t K, cudaStream_t stream) {
  ENGINE_CHECK(M >= 0 && N >= 0 && K >= 0, "matmul_naive: negative dimension");
  ENGINE_CHECK(A != nullptr && B != nullptr && C != nullptr,
               "matmul_naive: null device pointer");
  if (M == 0 || N == 0) return;

  // TODO(parag): remove this line and implement.
  //
  // Consider the K == 0 case: the product of an Mx0 and a 0xN matrix is the MxN zero
  // matrix, not undefined. cudaMemsetAsync is the tidy way to handle it.
  (void)stream;
  ENGINE_CHECK(false, "matmul_naive: not implemented yet (exercise 5)");
}

}  // namespace engine::cuda
