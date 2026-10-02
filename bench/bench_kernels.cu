//===----------------------------------------------------------------------===//
// bench/bench_kernels.cu -- the performance numbers that go in the report.
//
//   ./build/bin/bench_kernels                 # everything
//   ./build/bin/bench_kernels --quick         # fewer reps, for a fast sanity check
//   ./build/bin/bench_kernels > results.md    # paste straight into the notebook
//
// Read bench_harness.hpp before trusting anything this prints. In particular: LOCK
// THE GPU CLOCKS FIRST, and build with -DCMAKE_BUILD_TYPE=Release or RelWithDebInfo.
// A debug build turns on per-launch cudaDeviceSynchronize() and reports numbers that
// are wrong by an order of magnitude on small kernels.
//
// Unimplemented kernels appear as "--" rows rather than crashing the run, so this is
// useful from exercise 1 onward. The table filling in over the coming months IS the
// progress record -- commit each run into docs/lab-notebook.md with the date and the
// locked clock, and by June 2027 you will have a defensible optimisation history
// instead of a single final number.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#if !ENGINE_HAS_CUDA

#include <cstdio>
int main() {
  std::printf("bench_kernels: built without CUDA -- nothing to measure.\n");
  std::printf(
      "Run this on the GPU machine (RTX 4070 SUPER). For CPU baselines use "
      "bench_cpu_ref.\n");
  return 0;
}

#else

#include "bench_harness.hpp"

#include <engine/check.hpp>
#include <engine/cuda_device.hpp>
#include <engine/device_buffer.hpp>
#include <engine/kernels.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#if ENGINE_BENCH_CUBLAS
#include <cublas_v2.h>
#endif

namespace {

using engbench::Table;
using engine::DeviceBuffer;

/// Host-side random data. Filled once and reused: generating it is far slower than
/// the kernels being measured, and the VALUES do not affect timing for any of these
/// kernels -- there is no data-dependent branching on the GPU here. (That is worth
/// checking rather than assuming, though: a kernel with early exits on zeros WOULD
/// be data-dependent, and benchmarking it on all-zeros input would be meaningless.)
std::vector<float> random_host(std::size_t n, unsigned seed = 20260826u) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n);
  for (float& x : v) x = dist(gen);
  return v;
}

std::vector<engine::half> random_host_half(std::size_t n, unsigned seed = 20260826u) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<engine::half> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = engine::float_to_half(dist(gen));
  return v;
}

std::string mib(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f MiB",
                static_cast<double>(bytes) / (1024.0 * 1024.0));
  return buf;
}

double d(std::int64_t v) { return static_cast<double>(v); }

//===----------------------------------------------------------------------===//
// 1. vector_add -- the memory-bandwidth floor.
//
// 1 FLOP per 12 bytes moved (two reads, one write, 4 bytes each). Arithmetic
// intensity 0.083, against a balance point of ~82 on this card: this kernel is
// memory-bound by a factor of a thousand. The add itself is free.
//
// So there is exactly one question -- what fraction of peak DRAM bandwidth does it
// reach? -- and exactly one lever, which is whether the accesses coalesce. Anything
// above ~85% of peak is finished; there is no more performance in it. This is the
// benchmark that teaches you to look at the roofline BEFORE optimising, because
// nothing you do to the arithmetic can help.
//
// Small sizes are included on purpose: at n = 2^16 the kernel takes a few
// microseconds and the measurement is dominated by launch overhead, so achieved
// bandwidth looks terrible. That is not a kernel problem, it is a "the GPU is the
// wrong tool for 256 KiB" problem -- and knowing where that crossover sits matters
// when you decide which model layers are worth offloading.
//===----------------------------------------------------------------------===//
void bench_vector_add(Table& t, int reps) {
  const std::int64_t sizes[] = {1 << 16, 1 << 20, 1 << 24, 1 << 26};

  for (std::int64_t n : sizes) {
    const std::size_t un = static_cast<std::size_t>(n);
    const std::vector<float> h = random_host(un);
    DeviceBuffer<float> a(h), b(h), out(un);

    t.measure_gpu(
        "vector_add", mib(3 * un * sizeof(float)),
        /*flops=*/d(n),
        /*bytes=*/3.0 * d(n) * sizeof(float),
        [&] { engine::cuda::vector_add(a.get(), b.get(), out.get(), n); },
        /*warmup=*/5, reps);
  }
}

//===----------------------------------------------------------------------===//
// 2. reduce_sum -- bandwidth again, but with a synchronisation problem attached.
//
// Reads n floats, writes one. Ideal traffic is 4n bytes, so this should also run at
// peak bandwidth. It usually does not on a first attempt, and the reasons are the
// lesson: a reduction has to combine values ACROSS threads, which means shared
// memory, __syncthreads(), and a final cross-block step. Each of those is a chance to
// serialise.
//
// If this lands well below vector_add's GB/s, the kernel is spending its time
// synchronising rather than reading. The usual culprits, in order of how often they
// are the answer: only one thread per block doing the final combine, a tree that
// diverges within a warp, and a second kernel launch where __shfl_down_sync would do.
//===----------------------------------------------------------------------===//
void bench_reduce_sum(Table& t, int reps) {
  const std::int64_t sizes[] = {1 << 20, 1 << 24, 1 << 26};

  for (std::int64_t n : sizes) {
    const std::size_t un = static_cast<std::size_t>(n);
    DeviceBuffer<float> x(random_host(un));
    DeviceBuffer<float> out(1);

    t.measure_gpu(
        "reduce_sum", mib(un * sizeof(float)),
        /*flops=*/d(n),
        /*bytes=*/d(n) * sizeof(float),
        [&] { engine::cuda::reduce_sum(x.get(), out.get(), n); },
        /*warmup=*/5, reps);
  }
}

//===----------------------------------------------------------------------===//
// 3. softmax_rows -- the shape that actually matters.
//
// 8 x 50257 is not arbitrary: it is a batch of 8 logit vectors over a GPT-2-sized
// vocabulary, i.e. the softmax this engine will run once per generated token. If it
// is slow, every token is slow. Measure the thing you will actually do.
//
// 4096 x 4096 is the attention-score shape (a 4096-token sequence, one head), which
// is what FlashAttention replaces in February 2027. Keep this number: the headline
// claim of that module is a comparison against materialising this matrix, and you
// cannot make the comparison unless you measured the baseline first.
//
// FLOP count is approximate -- exp() is one "FLOP" here but costs several
// instructions -- so read the GB/s column, not GFLOP/s. Two full passes over the data
// (max, then exp-and-sum, then normalise) can in principle be fused into fewer; the
// bytes figure below assumes the ideal one-read-one-write, so a kernel that makes
// three passes shows up as achieving only a third of peak bandwidth. That is the
// intended signal, not an error in the accounting.
//===----------------------------------------------------------------------===//
void bench_softmax(Table& t, int reps) {
  struct Case {
    std::int64_t rows, cols;
    const char* label;
  };
  const Case cases[] = {
      {8, 50257, "8 x 50257  (GPT-2 logits)"},
      {128, 4096, "128 x 4096"},
      {4096, 4096, "4096 x 4096  (attention scores)"},
  };

  for (const Case& c : cases) {
    const std::size_t n = static_cast<std::size_t>(c.rows * c.cols);
    DeviceBuffer<float> in(random_host(n));
    DeviceBuffer<float> out(n);

    t.measure_gpu(
        "softmax_rows", c.label,
        /*flops=*/5.0 * d(c.rows * c.cols),
        /*bytes=*/2.0 * d(c.rows * c.cols) * sizeof(float),
        [&] { engine::cuda::softmax_rows(in.get(), out.get(), c.rows, c.cols); },
        /*warmup=*/5, reps);
  }
}

//===----------------------------------------------------------------------===//
// 4. rmsnorm -- 4096 columns because that is Llama-2-7B's hidden size.
//
// Same story as softmax: memory-bound, one read and one write per element, so the
// only interesting column is % of peak bandwidth. Included because RMSNorm runs
// twice per transformer layer (32 layers = 64 calls per token in Llama-2-7B), so even
// a kernel that is individually cheap is worth getting right -- and because it is the
// simplest kernel in the project that needs a per-row reduction, which makes it the
// natural warm-up for the softmax and attention kernels.
//===----------------------------------------------------------------------===//
void bench_rmsnorm(Table& t, int reps) {
  struct Case {
    std::int64_t rows, cols;
    const char* label;
  };
  const Case cases[] = {
      {1, 4096, "1 x 4096  (single token, Llama-2-7B)"},
      {512, 4096, "512 x 4096  (prefill batch)"},
      {4096, 4096, "4096 x 4096"},
  };

  for (const Case& c : cases) {
    const std::size_t n = static_cast<std::size_t>(c.rows * c.cols);
    DeviceBuffer<float> in(random_host(n));
    DeviceBuffer<float> w(random_host(static_cast<std::size_t>(c.cols), 7u));
    DeviceBuffer<float> out(n);

    t.measure_gpu(
        "rmsnorm", c.label,
        /*flops=*/4.0 * d(c.rows * c.cols),
        /*bytes=*/2.0 * d(c.rows * c.cols) * sizeof(float),
        [&] {
          engine::cuda::rmsnorm(in.get(), w.get(), out.get(), c.rows, c.cols, 1e-5f);
        },
        /*warmup=*/5, reps);
  }
}

//===----------------------------------------------------------------------===//
// 5. matmul -- naive vs tiled vs cuBLAS. THE HEADLINE TABLE OF OBJECTIVE 2.
//
// 2*M*N*K FLOPs against (MK + KN + MN)*4 bytes of ideal traffic. At 1024^3 that is
// arithmetic intensity 170, well above the ~82 balance point, so this is the first
// COMPUTE-BOUND kernel in the project -- and therefore the first one where being
// clever about arithmetic and data reuse can win anything at all.
//
// What the three rows per size are for:
//
//   naive   -- one thread per output element, each reading a full row of A and a full
//              column of B from DRAM. Every element of A is re-read N times. Its
//              ACTUAL traffic is ~2*M*N*K*4 bytes, not the ideal MK+KN+MN, so it is
//              memory-bound in practice despite being compute-bound in theory. This
//              gap between theoretical and actual traffic is the entire motivation
//              for tiling, and this row is the evidence for it.
//   tiled   -- each TILE of A and B is loaded into shared memory once and reused by
//              TILE threads, cutting DRAM traffic by a factor of TILE. Expect a large
//              speedup. Being able to predict its size from the tile dimension before
//              you measure it is the actual deliverable of exercise 6.
//   cuBLAS  -- the professional baseline. NOT competition: cuBLAS uses tensor cores,
//              hand-tuned assembly and shape-specific kernels. Reaching 40-60% of it
//              with a hand-written FP32 kernel is a genuinely good result and the
//              honest way to report your work. See ADR 0004 -- cuBLAS appears only
//              here, never as an implementation.
//
// 4096^3 is included because it is ~137 GFLOP of work: large enough that launch
// overhead is invisible and the number is purely about the kernel. Skipped under
// --quick because the naive kernel at that size takes a while.
//===----------------------------------------------------------------------===//
void bench_matmul(Table& t, int reps, bool include_large) {
  std::vector<std::int64_t> sizes = {512, 1024, 2048};
  if (include_large) sizes.push_back(4096);

  for (std::int64_t n : sizes) {
    const std::int64_t M = n, N = n, K = n;
    const std::size_t elems = static_cast<std::size_t>(n * n);
    const std::string label = std::to_string(n) + "^3";

    DeviceBuffer<float> A(random_host(elems, 1u));
    DeviceBuffer<float> B(random_host(elems, 2u));
    DeviceBuffer<float> C(elems);

    const double flops = 2.0 * d(M) * d(N) * d(K);
    const double bytes = (d(M) * d(K) + d(K) * d(N) + d(M) * d(N)) * sizeof(float);

    // Fewer reps at large sizes: the measurement is already stable (a 4096^3 naive
    // matmul takes long enough that scheduler noise is a rounding error) and 50 reps
    // of it would take minutes for no extra confidence.
    const int r = (n >= 2048) ? std::max(3, reps / 10) : reps;

    t.measure_gpu(
        "matmul_naive", label, flops, bytes,
        [&] { engine::cuda::matmul_naive(A.get(), B.get(), C.get(), M, N, K); },
        /*warmup=*/3, r);

    t.measure_gpu(
        "matmul_tiled", label, flops, bytes,
        [&] { engine::cuda::matmul_tiled(A.get(), B.get(), C.get(), M, N, K); },
        /*warmup=*/3, r);

    t.measure_gpu(
        "matmul_register_tiled", label, flops, bytes,
        [&] { engine::cuda::matmul_register_tiled(A.get(), B.get(), C.get(), M, N, K); },
        /*warmup=*/3, r);

#if ENGINE_BENCH_CUBLAS
    // CUBLAS IS COLUMN-MAJOR. This trips up everyone once.
    //
    // Our data is row-major. A row-major MxK matrix occupies the same bytes as a
    // column-major KxM matrix -- the interpretation changes, not the memory. So
    // reading our A as a cuBLAS matrix gives A^T, and likewise for B.
    //
    // We want row-major C = A*B. Using the identity C^T = B^T * A^T, and noting that
    // a column-major read of our row-major C IS C^T, the call becomes: ask cuBLAS for
    // (our B, read column-major = B^T) times (our A, read column-major = A^T), with
    // dimensions N, M, K in that order. No transposes and no copies -- CUBLAS_OP_N
    // for both operands. The leading dimensions are the row-major row lengths.
    //
    // The alternative (passing CUBLAS_OP_T) makes cuBLAS take a slower path, so this
    // argument-swapping trick is what a fair baseline requires, not just a tidy one.
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
      if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed; skipping cuBLAS baseline\n");
        handle = nullptr;
      }
    }
    if (handle != nullptr) {
      const float alpha = 1.0f, beta = 0.0f;
      const int in = static_cast<int>(n);
      t.measure_gpu(
          "cublasSgemm (baseline)", label, flops, bytes,
          [&] {
            const cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                  /*m=*/in, /*n=*/in, /*k=*/in, &alpha,
                                                  /*A=*/B.get(), /*lda=*/in,
                                                  /*B=*/A.get(), /*ldb=*/in, &beta,
                                                  /*C=*/C.get(), /*ldc=*/in);
            ENGINE_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");
          },
          /*warmup=*/3, r);
    }
#endif
  }
}

//===----------------------------------------------------------------------===//
// 6. Launch overhead -- the constant you need for every "is it worth it?" decision.
//
// An empty kernel launch costs a few microseconds. That is the FLOOR on any GPU
// operation, and it is why a 7B model's 32 layers cannot each afford ten separate
// kernel launches per token: at 5 us apiece, 320 launches is 1.6 ms of pure overhead
// per token, which caps you at ~600 tokens/s before any arithmetic happens.
//
// This measurement is the quantitative argument for kernel FUSION, which is what
// llama.cpp does aggressively and what your December 2026 work will need. Measure it
// once on this machine and write the number down.
//
// vector_add with n = 1 is the smallest real launch available without adding a
// dedicated empty kernel, so its median is launch overhead plus a rounding error.
//===----------------------------------------------------------------------===//
void bench_launch_overhead(Table& t, int reps) {
  DeviceBuffer<float> d1(1);
  d1.zero();
  t.measure_gpu(
      "vector_add (launch floor)", "n=1", /*flops=*/0.0, /*bytes=*/0.0,
      [&] { engine::cuda::vector_add(d1.get(), d1.get(), d1.get(), 1); },
      /*warmup=*/20, std::max(reps, 200));
}

//===----------------------------------------------------------------------===//
// 7. Module 3 -- Fused kernels: measuring the value of eliminating DRAM traffic.
//===----------------------------------------------------------------------===//
void bench_residual_rmsnorm(Table& t, int reps) {
  const struct {
    std::int64_t rows, cols;
    const char* label;
  } shapes[] = {
      {1, 4096, "1x4096 (decode token)"},
      {512, 4096, "512x4096 (prefill batch, warm L2)"},
      {4096, 4096, "4096x4096 (cold DRAM, 268 MB)"},
  };

  for (const auto& s : shapes) {
    const std::int64_t rows = s.rows;
    const std::int64_t cols = s.cols;
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    const std::string label = s.label;

    DeviceBuffer<float> x(random_host(n, 1u));
    DeviceBuffer<float> res(random_host(n, 2u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(cols), 3u));
    DeviceBuffer<float> norm_out(n);
    DeviceBuffer<float> sum_out(n);
    DeviceBuffer<float> temp_sum(n);

    const double flops = 5.0 * d(rows) * d(cols);
    const double separate_bytes = (5.0 * d(n) + d(cols)) * sizeof(float);
    const double fused_bytes = (4.0 * d(n) + d(cols)) * sizeof(float);

    t.measure_gpu(
        "residual+rmsnorm (separate)", label, flops, separate_bytes,
        [&] {
          engine::cuda::vector_add(x.get(), res.get(), temp_sum.get(), rows * cols);
          engine::cuda::rmsnorm(temp_sum.get(), weight.get(), norm_out.get(), rows, cols,
                                1e-5f);
        },
        /*warmup=*/5, reps);

    t.measure_gpu(
        "residual_rmsnorm (fused)", label, flops, fused_bytes,
        [&] {
          engine::cuda::residual_rmsnorm(x.get(), res.get(), weight.get(), norm_out.get(),
                                         sum_out.get(), rows, cols, 1e-5f);
        },
        /*warmup=*/5, reps);
  }
}

void bench_rmsnorm_linear(Table& t, int reps) {
  const struct {
    std::int64_t M, N, K;
  } shapes[] = {
      {1, 4096, 4096},    // single-token decode
      {512, 4096, 4096},  // prefill batch
      {1, 12288, 4096},   // QKV projection
  };

  for (const auto& s : shapes) {
    const std::int64_t M = s.M;
    const std::int64_t N = s.N;
    const std::int64_t K = s.K;
    const std::string label =
        std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

    DeviceBuffer<float> in(random_host(static_cast<std::size_t>(M * K), 1u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(K), 2u));
    DeviceBuffer<float> W(random_host(static_cast<std::size_t>(K * N), 3u));
    DeviceBuffer<float> out(static_cast<std::size_t>(M * N));
    DeviceBuffer<float> temp(static_cast<std::size_t>(M * K));

    const double flops = 4.0 * d(M) * d(K) + 2.0 * d(M) * d(N) * d(K);
    const double separate_bytes =
        (d(M) * d(K) * 2.0 + d(K) + d(K) * d(N) + d(M) * d(N)) * sizeof(float);
    const double fused_bytes =
        (d(M) * d(K) + d(K) + d(K) * d(N) + d(M) * d(N)) * sizeof(float);

    if (M == 1) {
      t.measure_gpu(
          "rmsnorm+gemv (fair baseline)", label, flops, separate_bytes,
          [&] {
            engine::cuda::rmsnorm(in.get(), weight.get(), temp.get(), M, K, 1e-5f);
            engine::cuda::gemv(W.get(), temp.get(), out.get(), N, K);
          },
          /*warmup=*/3, reps);
    }

    t.measure_gpu(
        "rmsnorm+matmul_register_tiled (unfused)", label, flops, separate_bytes,
        [&] {
          engine::cuda::rmsnorm(in.get(), weight.get(), temp.get(), M, K, 1e-5f);
          engine::cuda::matmul_register_tiled(temp.get(), W.get(), out.get(), M, N, K);
        },
        /*warmup=*/3, reps);

    t.measure_gpu(
        "rmsnorm+matmul (separate)", label, flops, separate_bytes,
        [&] {
          engine::cuda::rmsnorm(in.get(), weight.get(), temp.get(), M, K, 1e-5f);
          engine::cuda::matmul_tiled(temp.get(), W.get(), out.get(), M, N, K);
        },
        /*warmup=*/3, reps);

    t.measure_gpu(
        "rmsnorm_linear (1D fused)", label, flops, fused_bytes,
        [&] {
          engine::cuda::rmsnorm_linear_fused_direct(in.get(), weight.get(), W.get(),
                                                    out.get(), M, N, K, 1e-5f);
        },
        /*warmup=*/3, reps);

    t.measure_gpu(
        "rmsnorm_linear (dispatched)", label, flops, fused_bytes,
        [&] {
          engine::cuda::rmsnorm_linear(in.get(), weight.get(), W.get(), out.get(), M, N,
                                       K, 1e-5f);
        },
        /*warmup=*/3, reps);
  }
}

//===----------------------------------------------------------------------===//
// GEMV benchmarks (decode token generation: M=1, large K and N).
//===----------------------------------------------------------------------===//
void bench_gemv(Table& t, int reps) {
  struct Case {
    std::int64_t n, k;
    const char* label;
  };
  const Case cases[] = {
      {4096, 4096, "1x4096x4096 (decode token)"},
      {12288, 4096, "1x12288x4096 (decode MLP)"},
  };

  for (const Case& c : cases) {
    const std::int64_t N = c.n, K = c.k;
    const std::size_t matrix_elems = static_cast<std::size_t>(K * N);
    const std::size_t matrix_bytes = matrix_elems * sizeof(float);
    // Allocate enough distinct weight matrices to comfortably exceed the hardware L2 cache
    // (RTX 4070 SUPER has 48 MiB; RTX 4090 has 72 MiB enabled / 96 MiB physical).
    // For 4096^2 (64 MiB/matrix), 4 buffers = 256 MiB.
    // For 12288x4096 (192 MiB/matrix), 2 buffers = 384 MiB.
    const std::size_t num_buffers = (matrix_bytes >= 128 * 1024 * 1024) ? 2 : 4;
    std::vector<DeviceBuffer<float>> A_pool;
    A_pool.reserve(num_buffers);
    for (std::size_t b = 0; b < num_buffers; ++b) {
      A_pool.emplace_back(random_host(matrix_elems, static_cast<unsigned>(b + 1)));
    }

    DeviceBuffer<float> x(random_host(static_cast<std::size_t>(K), 2026u));
    DeviceBuffer<float> out(static_cast<std::size_t>(N));

    const double flops = 2.0 * d(N) * d(K);
    const double bytes = (d(K) * d(N) + d(K) + d(N)) * sizeof(float);

    // Warm-cache measurement for 4096^2 (for comparison against cold DRAM)
    if (N == 4096 && K == 4096) {
      t.measure_gpu(
          "gemv (warm L2)", c.label, flops, bytes,
          [&] { engine::cuda::gemv(A_pool[0].get(), x.get(), out.get(), N, K); },
          /*warmup=*/5, reps);
    }

    // Cold-cache measurements: rotating across the weight pool guarantees that each
    // iteration accesses a matrix evicted from L2 cache, measuring pure off-chip DRAM streaming.
    std::size_t iter_tiled = 0;
    t.measure_gpu(
        "matmul_tiled (M=1)", c.label, flops, bytes,
        [&] {
          const auto& A_cur = A_pool[iter_tiled % num_buffers];
          iter_tiled++;
          engine::cuda::matmul_tiled(x.get(), A_cur.get(), out.get(), 1, N, K);
        },
        /*warmup=*/5, reps);

    std::size_t iter_gemv = 0;
    t.measure_gpu(
        "gemv", c.label, flops, bytes,
        [&] {
          const auto& A_cur = A_pool[iter_gemv % num_buffers];
          iter_gemv++;
          engine::cuda::gemv(A_cur.get(), x.get(), out.get(), N, K);
        },
        /*warmup=*/5, reps);

    // FP16 GEMV (Cold DRAM streaming)
    {
      const std::size_t matrix_bytes_fp16 = matrix_elems * sizeof(engine::half);
      const std::size_t num_buffers_fp16 =
          (matrix_bytes_fp16 >= 128 * 1024 * 1024) ? 2 : 4;
      std::vector<DeviceBuffer<engine::half>> A_pool_fp16;
      A_pool_fp16.reserve(num_buffers_fp16);
      for (std::size_t b = 0; b < num_buffers_fp16; ++b) {
        A_pool_fp16.emplace_back(
            random_host_half(matrix_elems, static_cast<unsigned>(b + 100)));
      }
      DeviceBuffer<engine::half> x_fp16(
          random_host_half(static_cast<std::size_t>(K), 2026u));
      DeviceBuffer<engine::half> out_fp16(static_cast<std::size_t>(N));

      const double bytes_fp16 = (d(K) * d(N) + d(K) + d(N)) * sizeof(engine::half);

      // Warm L2 measurement for 4096^2
      if (N == 4096 && K == 4096) {
        t.measure_gpu(
            "gemv_fp16 (warm L2)", c.label, flops, bytes_fp16,
            [&] {
              engine::cuda::gemv_fp16(A_pool_fp16[0].get(), x_fp16.get(), out_fp16.get(),
                                      N, K);
            },
            /*warmup=*/5, reps);
      }

      std::size_t iter_gemv_fp16 = 0;
      t.measure_gpu(
          "gemv_fp16", c.label, flops, bytes_fp16,
          [&] {
            const auto& A_cur = A_pool_fp16[iter_gemv_fp16 % num_buffers_fp16];
            iter_gemv_fp16++;
            engine::cuda::gemv_fp16(A_cur.get(), x_fp16.get(), out_fp16.get(), N, K);
          },
          /*warmup=*/5, reps);
    }

#if ENGINE_BENCH_CUBLAS
    {
      cublasHandle_t handle;
      cublasCreate(&handle);
      const float alpha = 1.0f;
      const float beta = 0.0f;
      std::size_t iter_cublas = 0;
      t.measure_gpu(
          "cublasSgemm (M=1)", c.label, flops, bytes,
          [&] {
            const auto& A_cur = A_pool[iter_cublas % num_buffers];
            iter_cublas++;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N), 1,
                        static_cast<int>(K), &alpha, A_cur.get(), static_cast<int>(N),
                        x.get(), static_cast<int>(K), &beta, out.get(),
                        static_cast<int>(N));
          },
          /*warmup=*/5, reps);
      cublasDestroy(handle);
    }
#endif
  }
}

void bench_missing_kernels(Table& t, int reps) {
  // 1. Embedding lookup
  // Rotating across a pool of 16 random token ID buffers ensures iterations
  // do not artificially hit cached rows from preceding iterations.
  // Latency is reported rather than bandwidth.
  {
    const std::int64_t vocab = 32000;
    const std::int64_t hidden = 4096;
    const std::int64_t T = 512;
    const std::size_t num_token_pools = 16;

    std::vector<DeviceBuffer<std::int32_t>> d_ids_pool;
    d_ids_pool.reserve(num_token_pools);
    for (std::size_t b = 0; b < num_token_pools; ++b) {
      std::vector<std::int32_t> h_ids(static_cast<std::size_t>(T));
      std::mt19937 rng(static_cast<unsigned int>(42 + b));
      std::uniform_int_distribution<std::int32_t> dist(
          0, static_cast<std::int32_t>(vocab - 1));
      for (std::size_t i = 0; i < h_ids.size(); ++i) {
        h_ids[i] = dist(rng);
      }
      d_ids_pool.emplace_back(h_ids);
    }

    // Prefill: T = 512 (FP32)
    {
      std::vector<float> h_table = random_host(static_cast<std::size_t>(vocab * hidden));
      DeviceBuffer<float> d_table(h_table);
      DeviceBuffer<float> d_out(static_cast<std::size_t>(T * hidden));

      std::size_t iter_emb = 0;
      t.measure_gpu(
          "embedding (FP32)", "512 tokens (prefill)", /*flops=*/0.0, /*bytes=*/0.0,
          [&] {
            const auto& cur_ids = d_ids_pool[iter_emb % num_token_pools];
            iter_emb++;
            engine::cuda::embedding(d_table.get(), cur_ids.get(), d_out.get(), T, hidden,
                                    vocab);
          },
          /*warmup=*/5, reps);
    }

    // FP16 Prefill: T = 512
    {
      std::vector<engine::half> h_table =
          random_host_half(static_cast<std::size_t>(vocab * hidden));
      DeviceBuffer<engine::half> d_table(h_table);
      DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(T * hidden));

      std::size_t iter_emb_fp16 = 0;
      t.measure_gpu(
          "embedding_fp16", "512 tokens (prefill)", /*flops=*/0.0, /*bytes=*/0.0,
          [&] {
            const auto& cur_ids = d_ids_pool[iter_emb_fp16 % num_token_pools];
            iter_emb_fp16++;
            engine::cuda::embedding_fp16(d_table.get(), cur_ids.get(), d_out.get(), T,
                                         hidden, vocab);
          },
          /*warmup=*/5, reps);
    }
  }

  // 2. Argmax / greedy sampling
  // Latency is reported rather than bandwidth.
  {
    const std::int64_t vocab = 32000;
    std::vector<float> h_logits = random_host(static_cast<std::size_t>(vocab));
    DeviceBuffer<float> d_logits(h_logits);
    DeviceBuffer<std::int32_t> d_out(1);

    t.measure_gpu(
        "argmax (FP32)", "V=32000 (greedy sample)", /*flops=*/0.0, /*bytes=*/0.0,
        [&] { engine::cuda::argmax(d_logits.get(), d_out.get(), vocab); },
        /*warmup=*/5, reps);

    std::vector<engine::half> h_logits_fp16 =
        random_host_half(static_cast<std::size_t>(vocab));
    DeviceBuffer<engine::half> d_logits_fp16(h_logits_fp16);

    t.measure_gpu(
        "argmax_fp16", "V=32000 (greedy sample)", /*flops=*/0.0, /*bytes=*/0.0,
        [&] { engine::cuda::argmax_fp16(d_logits_fp16.get(), d_out.get(), vocab); },
        /*warmup=*/5, reps);
  }
}

//===----------------------------------------------------------------------===//
// Phase 4 -- Fused SwiGLU vs Unfused (SiLU + Mul).
//===----------------------------------------------------------------------===//

__global__ void silu_unfused_kernel(const float* in, float* out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = x / (1.0f + __expf(-x));
  }
}

__global__ void mul_unfused_kernel(const float* a, const float* b, float* out,
                                   int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i];
  }
}

void bench_swiglu(Table& t, int reps) {
  struct Case {
    std::int64_t n;
    const char* label;
  };
  const Case cases[] = {
      {11008, "1x11008  (decode MLP)"},
      {512 * 11008, "512x11008  (prefill MLP)"},
  };

  for (const Case& c : cases) {
    const std::int64_t n = c.n;
    DeviceBuffer<float> gate(random_host(static_cast<std::size_t>(n), 1u));
    DeviceBuffer<float> up(random_host(static_cast<std::size_t>(n), 2u));
    DeviceBuffer<float> temp_silu(static_cast<std::size_t>(n));
    DeviceBuffer<float> out(static_cast<std::size_t>(n));

    const double flops = 5.0 * d(n);
    const double unfused_bytes = 5.0 * d(n) * sizeof(float);
    const double fused_bytes = 3.0 * d(n) * sizeof(float);

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    // Warm the GPU to ramp SM and memory clocks into performance P0 state
    for (int w = 0; w < 30; ++w) {
      silu_unfused_kernel<<<blocks, threads>>>(gate.get(), temp_silu.get(), n);
      mul_unfused_kernel<<<blocks, threads>>>(temp_silu.get(), up.get(), out.get(), n);
      engine::cuda::swiglu(gate.get(), up.get(), out.get(), n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    t.measure_gpu(
        "swiglu (unfused: silu+mul)", c.label, flops, unfused_bytes,
        [&] {
          silu_unfused_kernel<<<blocks, threads>>>(gate.get(), temp_silu.get(), n);
          mul_unfused_kernel<<<blocks, threads>>>(temp_silu.get(), up.get(), out.get(),
                                                  n);
        },
        /*warmup=*/10, reps);

    t.measure_gpu(
        "swiglu (fused FP32)", c.label, flops, fused_bytes,
        [&] { engine::cuda::swiglu(gate.get(), up.get(), out.get(), n); },
        /*warmup=*/10, reps);

    std::vector<engine::half> h_gate_half =
        random_host_half(static_cast<std::size_t>(n), 1u);
    std::vector<engine::half> h_up_half =
        random_host_half(static_cast<std::size_t>(n), 2u);
    DeviceBuffer<engine::half> d_gate_half(h_gate_half);
    DeviceBuffer<engine::half> d_up_half(h_up_half);
    DeviceBuffer<engine::half> d_out_half(static_cast<std::size_t>(n));

    const double fused_bytes_fp16 = 3.0 * d(n) * sizeof(engine::half);
    t.measure_gpu(
        "swiglu_fp16 (fused FP16)", c.label, flops, fused_bytes_fp16,
        [&] {
          engine::cuda::swiglu_fp16(d_gate_half.get(), d_up_half.get(), d_out_half.get(),
                                    n);
        },
        /*warmup=*/5, reps);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  bool json_output = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else if (arg == "--json") {
      json_output = true;
    } else if (arg == "--help" || arg == "-h") {
      std::printf("usage: bench_kernels [--quick] [--json]\n");
      std::printf("  --quick   fewer repetitions, skips 4096^3 matmul\n");
      std::printf("  --json    output results as machine-readable JSON\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (engine::cuda_device_count() == 0) {
    std::fprintf(stderr,
                 "No CUDA device visible. This binary was built with CUDA support,\n"
                 "so the toolkit is installed but the driver sees no GPU. Check\n"
                 "`nvidia-smi`.\n");
    return 1;
  }

  if (!json_output) {
    engine::print_cuda_device_info();
  }

#if !defined(NDEBUG)
  std::fprintf(stderr,
               "\n*** WARNING: this is a DEBUG build. CUDA_CHECK_KERNEL() calls\n"
               "*** cudaDeviceSynchronize() after every launch, which dominates the\n"
               "*** time of any kernel shorter than a millisecond. Reconfigure with\n"
               "*** -DCMAKE_BUILD_TYPE=RelWithDebInfo before quoting these numbers.\n\n");
#endif

  const int reps = quick ? 10 : 50;

  Table t("Kernel benchmarks");
  bench_launch_overhead(t, reps);
  bench_vector_add(t, reps);
  bench_reduce_sum(t, reps);
  bench_softmax(t, reps);
  bench_rmsnorm(t, reps);
  bench_matmul(t, reps, /*include_large=*/!quick);
  bench_gemv(t, reps);
  bench_residual_rmsnorm(t, reps);
  bench_rmsnorm_linear(t, reps);
  bench_missing_kernels(t, reps);
  bench_swiglu(t, reps);

  if (json_output) {
    t.print_json(std::cout);
  } else {
    t.print();
    if (t.has_unimplemented()) {
      std::printf(
          "\nSome rows are blank because those kernels are still stubs. That is the\n"
          "expected state -- implement one, re-run, and watch a row appear. The HTML\n"
          "comment at the end of each blank row says which exercise it is waiting on.\n");
    }
  }
  return 0;
}

#endif  // ENGINE_HAS_CUDA
