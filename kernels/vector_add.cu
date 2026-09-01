//===----------------------------------------------------------------------===//
// kernels/vector_add.cu -- WORKED EXAMPLE. Read this before the other kernels.
//
// The maths is trivial on purpose (out[i] = a[i] + b[i]) so that nothing distracts
// from the CUDA execution model. Everything else in this project is a variation on
// the patterns below.
//===----------------------------------------------------------------------===//
//
// THE EXECUTION MODEL, IN THE ORDER IT MATTERS
//
// You launch a GRID of BLOCKS. Each block contains THREADS. You choose both sizes.
//
//     kernel<<<num_blocks, threads_per_block>>>(args...)
//
//   * A block runs entirely on ONE SM (streaming multiprocessor). The RTX 4090 has
//     128 SMs. Threads within a block can cooperate: they share fast on-chip shared
//     memory and can synchronise with __syncthreads().
//   * Threads in DIFFERENT blocks cannot synchronise at all during a kernel. That
//     restriction is what lets the GPU schedule blocks in any order, on any SM, and
//     is why a grid scales across hardware generations without recompiling.
//   * Threads execute in WARPS of 32. The 32 threads of a warp share one program
//     counter, so if they take different branches the warp executes both paths with
//     some threads masked off ("warp divergence"). Costly, and worth designing away.
//
// Each thread must work out which data element it owns. The canonical formula:
//
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//              ^which block   ^block size   ^position in block
//
// With blockDim.x = 256: block 0 covers indices 0..255, block 1 covers 256..511,
// and so on. Consecutive threads get consecutive indices, which is exactly what the
// memory system wants -- see the note on coalescing below.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {
namespace {

// 256 threads per block is the standard starting point:
//   * a multiple of the 32-wide warp, so no threads are wasted in a partial warp;
//   * small enough that many blocks fit per SM concurrently (good occupancy, which
//     is how the GPU hides memory latency -- while one warp waits on a load,
//     another computes);
//   * large enough to amortise per-block launch overhead.
// Treat it as a tunable, not a constant. Measure before changing it.
constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// Version 1: one thread per element. The obvious approach.
//===----------------------------------------------------------------------===//
__global__ void vector_add_simple(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, std::int64_t n) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // THE BOUNDS CHECK IS NOT OPTIONAL.
  // Blocks come in whole units, so for n = 1000 and blockDim = 256 you must launch
  // ceil(1000/256) = 4 blocks = 1024 threads. The last 24 threads have no element
  // to process. Without this guard they write past the end of `out` -- which is an
  // out-of-bounds write that may corrupt unrelated memory rather than crash.
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

//===----------------------------------------------------------------------===//
// Version 2: grid-stride loop. This is the one to use.
//===----------------------------------------------------------------------===//
//
// Each thread processes several elements, striding by the total thread count:
//
//     for (i = tid; i < n; i += total_threads)
//
// Why it is better than version 1:
//
//   * The grid size becomes independent of n. You can size the grid to fill the GPU
//     (e.g. a few blocks per SM) and it handles any n, including n larger than the
//     maximum grid dimension.
//   * Launch configuration stops being a function of the input, so the same launch
//     is reusable and tunable.
//   * Blocks are reused rather than created and retired, amortising setup cost.
//
// COALESCING: the stride pattern is chosen so that at every step the 32 threads of a
// warp read 32 CONSECUTIVE floats = 128 contiguous bytes. The memory controller
// services that as a small number of wide transactions. Had each thread instead
// taken a contiguous chunk (thread 0 gets 0..k, thread 1 gets k..2k), a warp would
// touch 32 addresses k floats apart, forcing 32 separate transactions and wasting
// most of the bytes fetched. Same total work, several times slower. This is the
// single most important performance idea in GPU programming.
//===----------------------------------------------------------------------===//
__global__ void vector_add_grid_stride(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ out, std::int64_t n) {
  const std::int64_t start =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(blockDim.x) * gridDim.x;  // total threads in grid

  for (std::int64_t i = start; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

//===----------------------------------------------------------------------===//
// A note on __restrict__
//
// It promises the compiler that these pointers do not alias -- that writing through
// `out` cannot change what `a` or `b` point at. Without it the compiler must
// conservatively reload a[i] and b[i] after every store to out[i], because they
// might overlap. With it, loads can be hoisted, batched and reordered. It is free
// performance, and it is a promise you must actually keep: never pass overlapping
// buffers to a kernel whose parameters are __restrict__.
//===----------------------------------------------------------------------===//

}  // namespace

void vector_add(const float* a, const float* b, float* out, std::int64_t n,
                cudaStream_t stream) {
  // Validate on the host, where a failure produces a readable error with a stack,
  // rather than inside the kernel where it becomes an illegal memory access.
  ENGINE_CHECK(n >= 0, "vector_add: n must be non-negative");
  if (n == 0) return;  // launching an empty grid is an error, so return early
  ENGINE_CHECK(a != nullptr && b != nullptr && out != nullptr,
               "vector_add: null device pointer");

  // Size the grid to fill the machine rather than to match n. 32 blocks per SM is a
  // generous oversubscription that gives the scheduler plenty of warps to hide
  // memory latency with; it is then capped so we never launch more blocks than
  // there is work for.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  const std::int64_t blocks_needed = (n + kBlockSize - 1) / kBlockSize;
  const std::int64_t blocks_wanted = static_cast<std::int64_t>(num_sms) * 32;
  const int grid = static_cast<int>(blocks_needed < blocks_wanted ? blocks_needed
                                                                 : blocks_wanted);

  vector_add_grid_stride<<<grid, kBlockSize, 0, stream>>>(a, b, out, n);

  // Catches launch-configuration errors immediately, and in debug builds also
  // synchronises to catch faults inside the kernel. See engine/check.hpp for why
  // this is necessary rather than paranoid.
  CUDA_CHECK_KERNEL();
}

//===----------------------------------------------------------------------===//
// PERFORMANCE REALITY CHECK -- do this arithmetic before optimising anything.
//
// Per element this kernel moves 12 bytes (read a, read b, write out; 4 bytes each)
// and performs 1 floating-point add. Its arithmetic intensity is therefore
//
//     1 FLOP / 12 bytes  ~=  0.083 FLOP/byte
//
// The RTX 4090 offers roughly 1008 GB/s of bandwidth and on the order of 80 TFLOP/s
// of FP32 throughput -- a ratio of about 80 FLOP per byte. Since 0.083 is three
// orders of magnitude below that, this kernel is hopelessly MEMORY BOUND. Its
// ceiling is
//
//     1008e9 bytes/s / 12 bytes per element  ~=  84 billion elements/s
//
// and no amount of cleverness in the arithmetic will beat that. The only lever is
// moving fewer bytes: fuse this operation into a neighbouring one so the
// intermediate never reaches global memory. Kernel fusion is the main optimisation
// available for elementwise work, and llama.cpp does it aggressively.
//
// Matmul is the opposite case -- O(n^3) work against O(n^2) data -- which is why
// tiling pays off there and not here. Exercise 6 is where that becomes concrete.
//===----------------------------------------------------------------------===//

}  // namespace engine::cuda
