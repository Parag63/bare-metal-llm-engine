//===----------------------------------------------------------------------===//
// kernels/reduce_sum.cu -- EXERCISE 2. Your implementation.
//
// Spec:      out[0] = sum(x[0..n))       (out points to 1 float of device memory)
// Oracle:    engine::cpu::reduce_sum
// Test:      build/bin/engine_tests --filter=kernels.reduce_sum
//              -> kernels.reduce_sum_matches_reference
//                 kernels.reduce_sum_of_empty_writes_zero
//
// WHY THIS IS THE MOST IMPORTANT EXERCISE
//
// vector_add needed no cooperation: thread i touched element i and finished. A
// reduction is the opposite -- every thread's partial result must be combined into
// one number, so threads must communicate. That is a genuinely different kind of
// problem, and the pattern you build here is reused directly by softmax (exercise
// 3), RMSNorm (exercise 4), and the row-max/row-sum machinery inside FlashAttention
// in January. Get it properly right and the rest come cheaply.
//===----------------------------------------------------------------------===//
//
// THE PROBLEM: threads in different blocks cannot synchronise. So a single kernel
// cannot reduce the whole array to one value in one clean step. Two standard ways
// around it:
//
//   (a) TWO-STAGE. Each block reduces its own chunk to one float, writing
//       partial[blockIdx.x]. Then either launch a second kernel to reduce the (now
//       tiny) partial array, or let block 0 finish the job. Deterministic, which
//       matters -- see the note at the bottom.
//
//   (b) ATOMIC FINISH. Each block reduces its chunk, then one thread per block does
//       atomicAdd(out, block_result). Simpler, one launch. But floating-point
//       addition is NOT associative, so the answer depends on the order blocks
//       happen to finish, and the result varies run to run. Fine here; a menace when
//       you are trying to determine whether a numerical difference is a real bug.
//
// Recommendation: implement (a). Then implement (b) as well and compare both speed
// and run-to-run reproducibility -- that comparison is a genuinely good paragraph
// for the report.
//
//===----------------------------------------------------------------------===//
// WITHIN A BLOCK, THREE TECHNIQUES, EACH FASTER THAN THE LAST
//
// 1. Shared-memory tree reduction.
//    Stage the block's elements in __shared__ float buf[kBlockSize], then halve the
//    active thread count each round:
//
//        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//          if (tid < s) buf[tid] += buf[tid + s];
//          __syncthreads();
//        }
//
//    Two traps. First, __syncthreads() must be reached by EVERY thread in the block
//    -- putting it inside `if (tid < s)` deadlocks. Second, the naive variant that
//    strides with `if (tid % (2*s) == 0)` looks equivalent but is much worse: it
//    leaves only every other thread active within each warp, so warps stay resident
//    doing nothing. The form above keeps active threads contiguous, so whole warps
//    retire together.
//
// 2. Warp-level shuffle for the last 32 elements.
//    Once s <= 32 the survivors are a single warp, which is implicitly synchronised,
//    so shared memory and __syncthreads() are both unnecessary:
//
//        for (int offset = 16; offset > 0; offset >>= 1)
//          v += __shfl_down_sync(0xffffffff, v, offset);
//
//    __shfl_down_sync reads a value directly from another lane's register. No memory
//    round trip at all. The mask 0xffffffff means "all 32 lanes participate" -- it
//    must be accurate, and every participating lane must execute the instruction.
//
// 3. Grid-stride accumulation before reducing.
//    Have each thread sum many elements into a local register first, exactly as in
//    vector_add, and only then reduce across threads. This is the biggest single win
//    for large n, because the expensive cross-thread part runs once instead of
//    once per chunk.
//
//===----------------------------------------------------------------------===//
// PITFALLS
//
//   * Forgetting to zero *out before accumulating into it (variant b).
//   * Assuming n is a multiple of the block size. Guard every load.
//   * Reading buf[tid + s] when tid + s >= blockDim.x on a partially filled block.
//     Pad the shared buffer with 0.0f instead of special-casing.
//   * A missing __syncthreads() between the write and the read of shared memory.
//     This produces results that are *usually* right, which is the worst outcome.
//     If a reduction is intermittently wrong, look here first.
//
// ACCEPTANCE
//   * Matches the CPU oracle within rtol 1e-5 for n in {1, 31, 32, 33, 255, 256,
//     257, 1<<20}. The awkward sizes are the point; a kernel that only works on
//     powers of two is not finished.
//   * n = 1<<24 sustains a large fraction of peak bandwidth. Reduction reads 4 bytes
//     per element and writes almost nothing, so bench/bench_kernels reporting
//     ~900 GB/s on the 4090 means you are done optimising.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {
namespace {

// 256 threads per block, matching vector_add. A multiple of the 32-wide warp, small
// enough for good occupancy, large enough to amortise per-block costs.
constexpr int kBlockSize = 256;

//===----------------------------------------------------------------------===//
// WITHIN-BLOCK REDUCTION: shared-memory tree + warp shuffle.
//
// This is a __device__ helper rather than inline code because the same reduction
// pattern is used by both the partial-sum kernel (stage 1) and the final kernel
// (stage 2). Factoring it out here means exercises 3 and 4 can reuse it too.
//
// Precondition: sdata[0..blockDim.x) is populated. tid = threadIdx.x.
// Postcondition: the sum is in sdata[0] / the return value of thread 0.
//===----------------------------------------------------------------------===//
__device__ float block_reduce_sum(float* sdata, int tid) {
  // Tree reduction in shared memory, halving active threads each round.
  // __syncthreads() is OUTSIDE the conditional -- putting it inside deadlocks,
  // because every thread in the block must reach it. This is the #1 pitfall
  // from the stub header.
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Once we are down to a single warp (32 threads), shared memory and
  // __syncthreads() are both unnecessary: warps are implicitly synchronous.
  // __shfl_down_sync reads directly from another lane's register -- no memory
  // round trip at all. The mask 0xffffffff means all 32 lanes participate.
  float val = sdata[tid];
  if (tid < 32) {
    // Pull in the upper-half contribution from shared memory one last time.
    // At this point s == 32, so the tree step that would have been
    // `if (tid < 32) sdata[tid] += sdata[tid + 32]` is done here explicitly.
    val += sdata[tid + 32];
    // Now reduce within the warp using shuffle intrinsics.
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }
  return val;
}

//===----------------------------------------------------------------------===//
// STAGE 1: each block reduces its chunk of x[] to one partial sum.
//
// 1. Grid-stride loop: each thread accumulates many elements into a local
//    register. This is the biggest single win for large n -- the expensive
//    cross-thread reduction runs once per block instead of once per chunk.
// 2. Store the per-thread sum into shared memory.
// 3. block_reduce_sum brings it down to one value in thread 0.
// 4. Thread 0 writes that value to partial[blockIdx.x].
//===----------------------------------------------------------------------===//
__global__ void reduce_sum_partial(const float* __restrict__ x,
                                   float* __restrict__ partial, std::int64_t n) {
  __shared__ float sdata[kBlockSize];

  const int tid = threadIdx.x;
  const std::int64_t start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + tid;
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

  // Grid-stride accumulation into a register. Every load is guarded: n is not
  // necessarily a multiple of the block size, and reading past the end is an
  // out-of-bounds access, not a rounding error.
  float thread_sum = 0.0f;
  for (std::int64_t i = start; i < n; i += stride) {
    thread_sum += x[i];
  }

  // Deposit into shared memory. Threads that had no work contribute 0.0f via
  // the initialised thread_sum, which is the padding the pitfalls section
  // requires -- no special-casing needed.
  sdata[tid] = thread_sum;
  __syncthreads();

  // Reduce within the block.
  const float val = block_reduce_sum(sdata, tid);

  // Thread 0 writes this block's partial sum.
  if (tid == 0) {
    partial[blockIdx.x] = val;
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2: reduce the (now tiny) partial[] array to a single scalar.
//
// Launched as a SINGLE block. The partial array has at most `grid` elements
// (typically a few thousand), which fits easily in one 256-thread block with a
// small loop. The result goes directly into out[0].
//===----------------------------------------------------------------------===//
__global__ void reduce_sum_final(const float* __restrict__ partial,
                                 float* __restrict__ out, int num_partials) {
  __shared__ float sdata[kBlockSize];

  const int tid = threadIdx.x;

  // Each thread accumulates its slice of the partial array. Guard every load --
  // num_partials is generally not a multiple of blockDim.x.
  float thread_sum = 0.0f;
  for (int i = tid; i < num_partials; i += static_cast<int>(blockDim.x)) {
    thread_sum += partial[i];
  }

  sdata[tid] = thread_sum;
  __syncthreads();

  const float val = block_reduce_sum(sdata, tid);

  if (tid == 0) {
    out[0] = val;
  }
}

}  // namespace

void reduce_sum(const float* x, float* out, std::int64_t n, cudaStream_t stream) {
  ENGINE_CHECK(n >= 0, "reduce_sum: n must be non-negative");
  ENGINE_CHECK(x != nullptr && out != nullptr, "reduce_sum: null device pointer");

  // The sum of nothing is 0. Write it explicitly -- leaving the buffer untouched
  // is wrong, because the caller cannot distinguish "your kernel declined to write"
  // from "the answer is whatever was in that memory". The test poisons the buffer
  // with -12345.0f to catch exactly this.
  if (n == 0) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    return;
  }

  // Size the grid to fill the machine, exactly as vector_add does: enough blocks
  // to keep the SMs busy, capped so we never launch more than there is work for.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  const std::int64_t blocks_needed = (n + kBlockSize - 1) / kBlockSize;
  const std::int64_t blocks_wanted = static_cast<std::int64_t>(num_sms) * 32;
  const int grid =
      static_cast<int>(blocks_needed < blocks_wanted ? blocks_needed : blocks_wanted);

  // Allocate a small temporary buffer for the per-block partial sums. At most
  // `grid` floats -- a few KB on any current GPU. This is the only allocation
  // reduce_sum makes, and it is freed before the function returns.
  float* partial = nullptr;
  CUDA_CHECK(cudaMalloc(&partial, static_cast<std::size_t>(grid) * sizeof(float)));

  // Stage 1: each block reduces its chunk of x[] to partial[blockIdx.x].
  reduce_sum_partial<<<grid, kBlockSize, 0, stream>>>(x, partial, n);
  CUDA_CHECK_KERNEL();

  // Stage 2: one block reduces partial[0..grid-1] to out[0].
  reduce_sum_final<<<1, kBlockSize, 0, stream>>>(partial, out, grid);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaFree(partial));
}

}  // namespace engine::cuda
