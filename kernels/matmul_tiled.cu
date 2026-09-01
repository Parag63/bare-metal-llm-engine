//===----------------------------------------------------------------------===//
// kernels/matmul_tiled.cu -- EXERCISE 6. Your implementation. The important one.
//
// Spec:      identical to matmul_naive -- C[MxN] = A[MxK] * B[KxN], row-major
// Oracle:    engine::cpu::matmul
// Test:      build/bin/engine_tests --filter=kernels.matmul
//              -> kernels.matmul_tiled_matches_reference
//                 kernels.matmul_tiled_agrees_with_matmul_naive
//                 kernels.matmul_with_k_zero_is_the_zero_matrix
//            The second one is the interesting test: it compares your two kernels to
//            each other at 17x23x31, where every edge tile is partial.
// Bench:     bench/bench_kernels reports naive vs tiled side by side
//
// Same maths, same output, no algorithmic change in the FLOP count. All that changes
// is WHERE THE DATA LIVES while it is being used. That is the whole lesson, and it is
// the lesson that carries into FlashAttention: FlashAttention is not a cheaper
// attention algorithm, it is the same attention with a better memory schedule.
//===----------------------------------------------------------------------===//
//
// THE GPU MEMORY HIERARCHY, WITH THE NUMBERS THAT MATTER
//
//   Registers      ~0 cycles      per-thread, a few hundred KB per SM in total
//   Shared memory  ~20-30 cycles  per-block, up to 99 KiB per block on sm_89
//   L2 cache       ~200 cycles    device-wide, 72 MB on the 4090
//   Global memory  ~400+ cycles   device-wide, 24 GB at ~1008 GB/s
//
// Shared memory is roughly an order of magnitude lower latency than global, it is
// explicitly managed rather than automatic, and it is shared by every thread in the
// block. Tiling means: cooperatively copy a block of data into shared memory once,
// then have all the threads that need it read it from there many times.
//
//===----------------------------------------------------------------------===//
// THE ALGORITHM
//
// Let TILE = 32 (a warp's width, and 32x32 floats = 4 KiB per tile).
// Each block computes one TILE x TILE patch of C, marching along the K dimension:
//
//   __shared__ float As[TILE][TILE];
//   __shared__ float Bs[TILE][TILE];
//
//   float acc = 0.0f;
//   for (int t = 0; t < ceil(K / TILE); ++t) {
//       load one TILE x TILE tile of A (columns t*TILE .. t*TILE+TILE) into As
//       load one TILE x TILE tile of B (rows    t*TILE .. t*TILE+TILE) into Bs
//       __syncthreads();                       // (1) tiles are fully written
//       for (int k = 0; k < TILE; ++k)
//           acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//       __syncthreads();                       // (2) everyone done reading
//   }
//   C[row * N + col] = acc;
//
// BOTH __syncthreads() ARE LOAD-BEARING AND FOR DIFFERENT REASONS.
//   (1) prevents a thread from reading a tile element another thread has not written.
//   (2) prevents a fast thread from starting iteration t+1 and overwriting shared
//       memory that a slower thread is still reading from iteration t.
// Omitting (2) is the classic bug: it passes small tests, then fails intermittently
// at larger sizes. If your tiled matmul is *usually* right, this is where to look.
//
//===----------------------------------------------------------------------===//
// WHY IT IS FASTER -- and know this number before you measure
//
// In the naive kernel each thread read 2K floats from global memory. Here, each block
// of TILE^2 = 1024 threads loads 2 * TILE * K floats in total and performs
// TILE^2 * 2K FLOPs. Global traffic per block falls by a factor of TILE:
//
//     naive:  0.25 FLOP/byte
//     tiled:  0.25 * 32  =  8 FLOP/byte
//
// Against the 4090's ~82 FLOP/byte balance point, 8 is still memory bound -- so
// expect a large speedup, not the ~30x the traffic reduction alone might suggest.
// Predict the number, then measure it, then explain the gap. That process is worth
// more than the number.
//
// The next step beyond tiling, if you want it: REGISTER BLOCKING, where each thread
// computes a 4x4 patch of C instead of one element, holding 16 accumulators in
// registers. That multiplies arithmetic intensity by another 4x and is how real GEMM
// kernels (and cuBLAS) reach peak. Optional, but a strong extension for the report,
// and December is the scheduled month for it.
//
//===----------------------------------------------------------------------===//
// SHARED MEMORY BANK CONFLICTS -- the detail that separates a good answer from a
// great one
//
// Shared memory is divided into 32 banks of 4 bytes, striped so that consecutive
// floats land in consecutive banks. A warp can service one access per bank per cycle.
// If two lanes in a warp hit DIFFERENT addresses in the SAME bank, the accesses
// serialise.
//
// In the inner loop, Bs[k][threadIdx.x] has consecutive lanes reading consecutive
// columns -> 32 different banks -> conflict-free. Good. But As[threadIdx.y][k] has
// every lane in a row reading the same address, which the hardware broadcasts for
// free. Also good. So the layout above is already clean -- but if you later transpose
// As to As[k][threadIdx.y], consecutive lanes read addresses TILE apart. With
// TILE = 32 that is a 32-way conflict: every access serialises 32 ways.
//
// The standard fix is padding: declare As[TILE][TILE + 1]. The extra column shifts
// each row by one bank so a column read spreads across all 32 banks. Costs 128 bytes
// of shared memory, recovers most of the loss. Try the conflicted version, measure
// it, add the padding, measure again -- Nsight Compute reports bank conflicts
// directly, and that before/after pair is an excellent figure for the report.
//
// PITFALLS
//   * Not zero-padding tiles at the matrix edges. When K is not a multiple of TILE the
//     last tile is partial; load 0.0f into the out-of-range slots rather than
//     skipping the load, so the inner product loop needs no special case.
//   * Bounds-checking the tile LOAD but not the final store to C, or vice versa.
//   * Assuming blockDim equals TILE in both dimensions. Either assert it or derive
//     the indices properly -- do not leave it implicit.
//   * Declaring shared arrays inside the k-loop. They must be outside it.
//
// ACCEPTANCE
//   * Passes exactly the same test suite as matmul_naive, including (17,23,31).
//     If it passes the square cases but not that one, your edge padding is wrong.
//   * A clear, measured speedup over naive at 1024^3, with a written explanation of
//     why the measured factor differs from the predicted 32x.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {
namespace {

// 32 is the natural first choice: it matches the warp width, and two 32x32 float
// tiles need 8 KiB of the 99 KiB available per block on sm_89 -- leaving ample room
// for several blocks to be resident per SM, which is what keeps occupancy high.
// Treat it as a tunable. Try 16 and 64 and record what happens.
// constexpr int kTile = 32;

// TODO(parag): your kernel here.
//
// __global__ void matmul_tiled_kernel(const float* __restrict__ A,
//                                     const float* __restrict__ B,
//                                     float* __restrict__ C,
//                                     std::int64_t M, std::int64_t N,
//                                     std::int64_t K) { ... }

}  // namespace

void matmul_tiled(const float* A, const float* B, float* C, std::int64_t M,
                  std::int64_t N, std::int64_t K, cudaStream_t stream) {
  ENGINE_CHECK(M >= 0 && N >= 0 && K >= 0, "matmul_tiled: negative dimension");
  ENGINE_CHECK(A != nullptr && B != nullptr && C != nullptr,
               "matmul_tiled: null device pointer");
  if (M == 0 || N == 0) return;

  // TODO(parag): remove this line and implement.
  (void)stream;
  ENGINE_CHECK(false, "matmul_tiled: not implemented yet (exercise 6)");
}

}  // namespace engine::cuda
