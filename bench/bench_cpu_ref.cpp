//===----------------------------------------------------------------------===//
// bench/bench_cpu_ref.cpp -- the CPU baseline. THE DENOMINATOR.
//
//   ./build/bin/bench_cpu_ref
//   ./build/bin/bench_cpu_ref --quick
//
// Every "the GPU kernel is Nx faster" claim in the report is a fraction, and this
// binary measures the bottom of it. Two reasons that matters more than it sounds:
//
//   1. AN UNSPECIFIED BASELINE IS NOT A RESULT. "40x faster than CPU" means nothing
//      without saying which CPU, which compiler, which flags, single-threaded or not,
//      and which implementation. The table this prints records all of that, so the
//      speedup is a number someone else could reproduce or dispute.
//   2. IT IS AN HONEST BASELINE, AND YOU SHOULD SAY SO. src/cpu_ref/ is written for
//      obvious correctness -- scalar loops, no SIMD, no blocking, no threads, FP64
//      accumulators. It is therefore SLOWER than a competent CPU implementation, very
//      much so for matmul, where a blocked AVX2 multi-threaded version is easily
//      50-100x this code. Comparing a tuned GPU kernel against a deliberately naive
//      CPU one inflates the speedup, and a mentor or interviewer will ask about it.
//      Report the number, then say what it is measured against. That framing is
//      stronger than the inflated figure, because it shows you know the difference.
//
// This builds and runs WITHOUT a GPU, which is the point: it is the benchmark you can
// run on the laptop, and the CPU numbers do not change when you move machines.
//===----------------------------------------------------------------------===//

#include "bench_harness.hpp"

#include <engine/cpu_ref.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using engbench::Table;

std::vector<float> random_host(std::size_t n, unsigned seed = 20260826u) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n);
  for (float& x : v) x = dist(gen);
  return v;
}

double d(std::int64_t v) { return static_cast<double>(v); }

std::string mib(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f MiB",
                static_cast<double>(bytes) / (1024.0 * 1024.0));
  return buf;
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
      std::printf("usage: bench_cpu_ref [--quick] [--json]\n");
      std::printf("  --quick   fewer repetitions, skips the 1024^3 matmul\n");
      std::printf("  --json    output results as machine-readable JSON\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

#if !defined(NDEBUG)
  std::fprintf(stderr,
               "\n*** WARNING: DEBUG build. These loops are 5-20x slower without\n"
               "*** optimisation, which would make every GPU speedup look far\n"
               "*** better than it is. Rebuild with -DCMAKE_BUILD_TYPE=Release.\n\n");
#endif

  const int reps = quick ? 3 : 10;
  Table t("CPU reference baselines (single-threaded scalar, FP64 accumulators)");

  // --- vector_add -----------------------------------------------------------
  // Memory-bound on the CPU too, but against a much lower ceiling: dual-channel
  // DDR4/DDR5 is roughly 50-100 GB/s against the 4090's ~1008. That ratio alone --
  // about 10-20x -- is the MOST a memory-bound kernel can ever gain from the GPU, no
  // matter how good the kernel is. Worth knowing before spending a week on one.
  for (std::int64_t n : {std::int64_t{1} << 20, std::int64_t{1} << 24}) {
    const std::size_t un = static_cast<std::size_t>(n);
    const std::vector<float> a = random_host(un, 1u);
    const std::vector<float> b = random_host(un, 2u);
    std::vector<float> out(un);
    t.measure_cpu(
        "cpu::vector_add", mib(3 * un * sizeof(float)), d(n), 3.0 * d(n) * sizeof(float),
        [&] { engine::cpu::vector_add(a.data(), b.data(), out.data(), n); },
        /*warmup=*/2, reps);
  }

  // --- reduce_sum -----------------------------------------------------------
  // A sequential accumulation with a loop-carried dependency: each add must wait for
  // the previous one, so this runs at roughly one add per FP-add latency (~4 cycles)
  // rather than at memory speed. The GPU's advantage here is not just bandwidth, it
  // is that a tree reduction has no such dependency chain -- the same structural
  // reason it is also MORE ACCURATE. (See tree_reduction_atol in tests/test_kernels.cu.)
  {
    const std::int64_t n = std::int64_t{1} << 24;
    const std::vector<float> x = random_host(static_cast<std::size_t>(n));
    volatile double sink = 0.0;  // stops the optimiser deleting the whole call
    t.measure_cpu(
        "cpu::reduce_sum", mib(static_cast<std::size_t>(n) * sizeof(float)), d(n),
        d(n) * sizeof(float), [&] { sink = engine::cpu::reduce_sum(x.data(), n); },
        /*warmup=*/2, reps);
    (void)sink;
  }

  // --- softmax_rows ---------------------------------------------------------
  // Dominated by std::exp, which is a library call of tens of cycles each. This is
  // where the GPU's special-function units earn their keep: expf runs on dedicated
  // hardware at a rate of one per cycle per SFU, so the gap here is much larger than
  // the bandwidth ratio would suggest.
  {
    const std::int64_t rows = 8, cols = 50257;
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    const std::vector<float> in = random_host(n);
    std::vector<float> out(n);
    t.measure_cpu(
        "cpu::softmax_rows", "8 x 50257  (GPT-2 logits)", 5.0 * d(rows * cols),
        2.0 * d(rows * cols) * sizeof(float),
        [&] { engine::cpu::softmax_rows(in.data(), out.data(), rows, cols); },
        /*warmup=*/2, reps);
  }

  // --- rmsnorm --------------------------------------------------------------
  {
    const std::int64_t rows = 512, cols = 4096;
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    const std::vector<float> in = random_host(n);
    const std::vector<float> w = random_host(static_cast<std::size_t>(cols), 7u);
    std::vector<float> out(n);
    t.measure_cpu(
        "cpu::rmsnorm", "512 x 4096  (Llama-2-7B hidden)", 4.0 * d(rows * cols),
        2.0 * d(rows * cols) * sizeof(float),
        [&] { engine::cpu::rmsnorm(in.data(), w.data(), out.data(), rows, cols, 1e-5f); },
        /*warmup=*/2, reps);
  }

  // --- matmul ---------------------------------------------------------------
  // The headline comparison, and the one to be most careful about quoting. This is a
  // triple loop with a double accumulator and no blocking: it misses cache on nearly
  // every access to B and cannot use FMA on the FP64 accumulator. Expect single-digit
  // GFLOP/s against a CPU peak in the hundreds.
  //
  // So when you report the GPU speedup for matmul, give two numbers: against THIS
  // baseline, and against cuBLAS (which bench_kernels measures). The first says how
  // much the GPU is worth; the second says how good your kernel is. Only quoting the
  // first would be the kind of comparison this project is meant to teach you to
  // distrust.
  //
  // 512^3 is 268 MFLOP and takes a fraction of a second. 1024^3 is 2.1 GFLOP and
  // takes a few seconds here, hence --quick.
  {
    std::vector<std::int64_t> sizes = {256, 512};
    if (!quick) sizes.push_back(1024);
    for (std::int64_t n : sizes) {
      const std::size_t elems = static_cast<std::size_t>(n * n);
      const std::vector<float> A = random_host(elems, 1u);
      const std::vector<float> B = random_host(elems, 2u);
      std::vector<float> C(elems);
      t.measure_cpu(
          "cpu::matmul", std::to_string(n) + "^3", 2.0 * d(n) * d(n) * d(n),
          (3.0 * d(n) * d(n)) * sizeof(float),
          [&] { engine::cpu::matmul(A.data(), B.data(), C.data(), n, n, n); },
          /*warmup=*/1, std::max(3, reps / 2));
    }
  }

  if (json_output) {
    t.print_json(std::cout);
  } else {
    t.print();
    std::printf(
        "\nWhen you quote a speedup against these numbers, say what the baseline is:\n"
        "  \"Nx faster than a single-threaded scalar C++ reference implementation\"\n"
        "and not \"Nx faster than CPU\". The first is a fact; the second is a claim you\n"
        "cannot defend, because a blocked multi-threaded AVX2 matmul would be far\n"
        "closer to the GPU than this code is.\n");
  }
  return 0;
}
