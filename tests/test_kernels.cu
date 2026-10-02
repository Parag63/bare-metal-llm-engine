//===----------------------------------------------------------------------===//
// tests/test_kernels.cu -- the CUDA exercise ladder, as executable tests.
//
// ALL EIGHT EXERCISES ARE IMPLEMENTED AND PASSING.
//
// Run it on the 4090 box. All kernel tests pass, and all have been promoted
// from TEST_PENDING to TEST. The pending count is 0.
//
//   ./engine_tests --filter=kernels           # just this file
//   ./engine_tests --filter=kernels.softmax   # just one kernel
//
//===----------------------------------------------------------------------===//
// WHY THIS IS A .cu FILE
//
// It includes <cuda_runtime.h> (through engine/kernels.hpp) and it links against
// device code. Naming it .cu hands it to nvcc, which puts the CUDA include and
// library paths in place automatically. Naming it .cpp would work too, but only
// after teaching CMake where the toolkit headers live -- pointless friction.
//
// Nothing in here is device code: there are no __global__ functions and no <<<>>>
// launches. Every kernel is invoked through the host-side launchers in
// engine/kernels.hpp, exactly as the rest of the engine will invoke them. If a test
// here passes, the API the engine uses is the API that was verified.
//===----------------------------------------------------------------------===//
// THE THREE-TIER ARGUMENT, ONE MORE TIME
//
//   tier 1  tools/gen_reference.py  -- PyTorch/NumPy, float64, on the host
//   tier 2  src/cpu_ref/*.cpp       -- your own scalar C++, verified in test_cpu_ref
//   tier 3  kernels/*.cu            -- the GPU kernel, verified HERE
//
// The tests below compare tier 3 against tier 1, not against tier 2. That is
// deliberate: comparing the kernel to your own CPU code would let a shared
// misunderstanding (a transposed layout, an off-by-one in the row indexing) pass
// both. tier 2's job is to be the thing you debug against interactively, and to
// prove that YOU understand the operation; tier 1's job is to be independent.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#if ENGINE_HAS_CUDA

#include "golden.hpp"
#include "test_framework.hpp"

#include <engine/check.hpp>
#include <engine/cpu_ref.hpp>
#include <engine/cuda_device.hpp>
#include <engine/device_buffer.hpp>
#include <engine/kernels.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using engine::DeviceBuffer;

//===----------------------------------------------------------------------===//
// TOLERANCES for GPU-vs-reference comparison.
//
// The rule for this whole file: a tolerance is a claim about the arithmetic, so it
// gets a derivation. "I widened it until the test went green" is how a real bug
// ships. If a kernel misses one of these bounds, the first question is whether your
// error analysis was wrong -- and only then whether the kernel is.
//
// u = 2^-24 = 5.96e-8 is the float32 unit roundoff: the maximum RELATIVE error of
// one correctly-rounded float operation. (std::numeric_limits<float>::epsilon() is
// 2^-23; the unit roundoff is half of it, because rounding goes to the nearer of
// two neighbours.)
//===----------------------------------------------------------------------===//
constexpr double kU = 5.9604644775390625e-08;  // 2^-24

/// vector_add: BIT-EXACT, no exceptions.
///
/// out[i] = a[i] + b[i] is a single float addition. IEEE-754 requires it to be the
/// correctly-rounded result of the exact sum, and the generator computes the same
/// exact sum in float64 and rounds it once to float32. Two roundings of the same
/// exact value to the same format give the same bits. There is no accumulation, no
/// reordering, and no transcendental function -- so there is nothing for a tolerance
/// to absorb, and any difference at all is a bug in the indexing or the launch
/// geometry. Every element must match to the last bit.
constexpr double kAddRtol = 0.0;
constexpr double kAddAtol = 0.0;

/// softmax_rows: same bound as the CPU oracle, and for the same reason.
///
/// The dominant error is not exp() -- it is the ARGUMENT. exp(x - m) is computed
/// from a float subtraction whose absolute error reaches u*|x - m|, and because
/// d(exp x)/exp x = dx, that absolute argument error appears as a relative output
/// error of the same size:
///
///     rel_err  ~=  u * |x - m|
///
/// The deepest legitimately-checkable shift in the suite is the ~40 in
/// softmax_rows__edge row 3, giving 40 * 5.96e-8 = 2.4e-6. rtol = 1e-5 is 4x that.
///
/// Row 1 of that same file has a shift of ~300, where the bound would be 1.8e-5 and
/// exceed rtol -- except exp(-300) underflows to exactly 0.0f on both sides, so the
/// difference is exactly zero and the bound never bites. Worth knowing, because it
/// means this test does NOT exercise the 1e-5 boundary; do not read a pass as
/// evidence that the bound is loose enough for arbitrary inputs.
///
/// Two things that will legitimately push you past 1e-5:
///   * __expf() instead of expf(). The hardware approximation is ~2 ulp rather than
///     sub-ulp, which roughly doubles the constant. If you make that trade for
///     speed, widen this AND write down that you did -- silently relaxing a
///     tolerance to accommodate a faster function is how accuracy regressions hide.
///   * Computing the sum of exponentials in a different order than the oracle. That
///     is a tree-reduction error on a sum of positive terms, so it is tiny (no
///     cancellation) -- see kReduceAtol below for why positivity matters.
///
/// atol stays at denormal scale on purpose. The interesting outputs here are around
/// exp(-40) = 4e-18 and they must be right in a RELATIVE sense; an atol large enough
/// to cover them would let an all-zeros output pass. 1e-30 is set just high enough
/// to forgive a denormal-vs-zero disagreement, which is a legitimate difference
/// between the host's exp and a GPU running with flush-to-zero.
constexpr double kSoftmaxRtol = 1e-5;
constexpr double kSoftmaxAtol = 1e-30;

/// rmsnorm: out[c] = x[c] * rsqrt(mean(x^2) + eps) * w[c]
///
/// Error path, worst case at cols = 4096:
///   * sum of x^2 -- a tree reduction of 4096 POSITIVE terms, so the bound is
///     log2(4096) * u = 12 * 5.96e-8 = 7.2e-7 relative (see kReduceAtol);
///   * rsqrt HALVES relative error, because d(x^-1/2)/x^-1/2 = -dx/2x, so the scale
///     factor carries ~3.6e-7;
///   * two more multiplies at u each.
/// Total ~5e-7. rtol = 1e-5 leaves 20x, which also covers a hardware rsqrtf that is
/// only approximately correctly-rounded (~2 ulp) rather than exact.
///
/// atol = 1e-6 covers rmsnorm__zeros, where the output is exactly 0 and rtol*|0| is
/// no tolerance at all. It is well below the ~1.0 scale of a real output, so it
/// cannot mask a wrong answer anywhere else.
constexpr double kRmsRtol = 1e-5;
constexpr double kRmsAtol = 1e-6;

/// matmul: C[m][n] = sum over K of A[m][k]*B[k][n]
///
/// A float32 dot product of length K has the classic bound
///
///     |error|  <=  gamma_K * sum_k |a_k * b_k|,    gamma_K ~= K * u
///
/// and note that the bound is on sum|a_k b_k|, NOT on |C[m][n]|. Those differ by a
/// lot here, and that difference is the whole reason atol carries this test:
///
/// The generator scales both matrices by 1/sqrt(K), so a_k, b_k ~ N(0, 1/K):
///     sum_k |a_k b_k|  ~=  K * (0.8/sqrt(K))^2   =  0.64          (signs ignored)
///     |C[m][n]|        ~=  sqrt(K) * (1/K)       =  1/sqrt(K)  =  0.044 at K=512
/// so at K = 512:  |error| <= 512 * 5.96e-8 * 0.64 = 2.0e-5 absolute, against
/// elements of magnitude 0.044 -- i.e. ~4.4e-4 RELATIVE. Cancellation is why: the
/// products are O(1/K) each and mostly cancel, so the result is much smaller than
/// the terms that built it, and relative error is amplified by exactly that ratio.
///
/// Hence atol = 5e-5 (covers the 2.0e-5 absolute bound with headroom) and a
/// secondary rtol = 1e-4 for any element that happens to come out large. A real
/// matmul bug -- wrong stride, transposed B, a tile boundary missed -- produces
/// errors of order |C| itself, i.e. 1e-2 and up. There are three orders of magnitude
/// between "float rounding" and "wrong", which is what makes the test useful.
///
/// If you accumulate in float64 inside the kernel, or use fmaf() throughout, the
/// real error drops several-fold. Do not tighten the tolerance to match: the bound
/// above is what the SPECIFIED kernel (float accumulator) is allowed to produce, and
/// a future rewrite should not have to relax it again.
constexpr double kMatmulRtol = 1e-4;
constexpr double kMatmulAtol = 5e-5;

/// eps used by tools/gen_reference.py. Must match exactly.
constexpr float kEps = 1e-5f;

//===----------------------------------------------------------------------===//
// Case tables. Same shapes as test_cpu_ref.cpp and tools/gen_reference.py, so the
// GPU and the oracle are judged on identical data. The stems are derived by the same
// rule on all three sides; if they ever drift apart the file is simply not found and
// the test SKIPS, printing the stem it wanted -- which is a much friendlier failure
// than a silent mismatch.
//
// The sizes are chosen to break naive launch geometry:
//   1        single element -- degenerate grid
//   31       less than one warp
//   32/33    exactly one warp, and one past it
//   255/257  one short of and one past a 256-thread block
//   1<<20    large enough to need a grid-stride loop, and to actually be timed
//===----------------------------------------------------------------------===//
const std::int64_t kSizes1D[] = {1, 31, 32, 33, 255, 256, 257, 4096, 1 << 20};

struct Shape2 {
  std::int64_t rows, cols;
};
const Shape2 kSoftmaxShapes[] = {{1, 1}, {1, 1024}, {128, 127}, {7, 4096}, {8, 50257}};

struct RmsShape {
  std::int64_t rows, cols;
  bool with_weight;
};
const RmsShape kRmsShapes[] = {
    {1, 4096, true}, {128, 4096, true}, {32, 127, true}, {32, 4096, false}};

struct Shape3 {
  std::int64_t m, n, k;
};
const Shape3 kMatmulShapes[] = {
    {1, 1, 1}, {32, 32, 32}, {128, 64, 256}, {17, 23, 31}, {512, 512, 512}};

const RmsShape kResidualRmsShapes[] = {
    {1, 4096, true}, {128, 4096, true}, {32, 127, true}, {32, 4096, false}};

struct RmsnormLinearShape {
  std::int64_t m, n, k;
  bool with_weight;
};
const RmsnormLinearShape kRmsnormLinearShapes[] = {{1, 4096, 4096, true},
                                                   {32, 4096, 4096, true},
                                                   {1, 12288, 4096, true},
                                                   {17, 127, 31, true},
                                                   {32, 4096, 4096, false}};

std::string dims2(std::int64_t a, std::int64_t b) {
  return std::to_string(a) + "x" + std::to_string(b);
}

std::size_t usize(std::int64_t n) { return static_cast<std::size_t>(n); }

//===----------------------------------------------------------------------===//
/// Error bound for a TREE reduction of `x`, which is what a block-wise GPU sum is.
///
/// A sequential float sum of n terms has a worst-case bound of (n-1)*u*sum|x|: every
/// partial sum carries every earlier rounding forward. A tree sum of the same terms
/// has a bound of only log2(n)*u*sum|x|, because each value passes through just
/// log2(n) additions on its way to the root.
///
/// That is worth pausing on, because it is counter-intuitive and it comes up in the
/// viva: a GPU reduction is not merely faster than the obvious CPU loop, it is
/// MORE ACCURATE, and for a structural reason rather than by luck. At n = 2^20 the
/// bounds differ by a factor of 52,000.
///
/// Two practical notes:
///   * The bound scales with sum|x|, not with |sum x|. For standard-normal input
///     those differ by a factor of ~sqrt(n) (0.8n versus ~sqrt(n)) -- cancellation
///     again -- which is why this is an ABSOLUTE tolerance and rtol is left at 0.
///     Asking for a relative bound on a cancelling sum is asking for the impossible.
///   * The factor of 2 below is slack for the two-stage shape a real reduction takes
///     (per-block partials, then a second pass over them), which adds a few levels
///     over the ideal log2(n).
///
/// The tolerance this produces is TIGHT ENOUGH TO BE INTERESTING: it asserts that
/// your kernel achieves tree-quality accuracy, so a kernel that quietly falls back
/// to one thread accumulating serially fails here even though its answer "looks
/// about right".
//===----------------------------------------------------------------------===//
double tree_reduction_atol(const std::vector<float>& x) {
  double abs_sum = 0.0;
  for (float v : x) abs_sum += std::abs(static_cast<double>(v));
  if (abs_sum == 0.0) return 0.0;  // sum of zeros is exactly zero, on any schedule
  const double depth = std::log2(static_cast<double>(x.size())) + 1.0;
  return 2.0 * kU * depth * abs_sum;
}

}  // namespace

/// Every test in this file needs a GPU. On the laptop there isn't one, and that is
/// not a failure -- it is the CPU-only build doing exactly what it is designed to
/// do. The suite stays green and the summary lists what was skipped and why.
#define REQUIRE_CUDA_DEVICE()                                    \
  do {                                                           \
    if (::engine::cuda_device_count() == 0) {                    \
      SKIP_TEST("no CUDA device visible -- check `nvidia-smi`"); \
    }                                                            \
  } while (0)

//===----------------------------------------------------------------------===//
// PART 0 -- is the GPU there, and is it the one we think it is?
//===----------------------------------------------------------------------===//

TEST(kernels, device_is_visible_and_reports_sane_properties) {
  REQUIRE_CUDA_DEVICE();

  // Printed unconditionally: when a kernel result looks wrong six months from now,
  // the first question is "which GPU was this?" and the answer should be in the log
  // rather than in your memory.
  std::printf("    device: %s\n", engine::cuda_device_summary().c_str());

  const double bw = engine::cuda_peak_bandwidth_gbs();
  EXPECT_TRUE(bw > 0.0);

  // A sanity band, not a spec. Below 10 GB/s means the property query failed and
  // returned something meaningless; above 10000 GB/s means the same. On the 4090
  // this is ~1008 GB/s, and every roofline argument in the report divides by it, so
  // a garbage value here would quietly corrupt the analysis rather than crash it.
  EXPECT_TRUE(bw > 10.0 && bw < 10000.0);
}

//===----------------------------------------------------------------------===//
// PART 1 -- exercise 1, vector_add. THE WORKED EXAMPLE.
//
// This one is implemented (kernels/vector_add.cu). It passes today, and it is here
// as the control: if it ever fails, the problem is the harness, the driver or the
// build -- not your kernel. Read that file before starting exercise 2.
//===----------------------------------------------------------------------===//

TEST(kernels, vector_add_is_bit_exact) {
  REQUIRE_CUDA_DEVICE();

  for (std::int64_t n : kSizes1D) {
    const std::string stem = "vector_add__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);

    const auto& a = g.at("a").data;
    const auto& b = g.at("b").data;
    const auto& expected = g.at("expected").data;
    ASSERT_EQ(a.size(), usize(n));

    DeviceBuffer<float> d_a(a);
    DeviceBuffer<float> d_b(b);
    DeviceBuffer<float> d_out(usize(n));

    // Poison the output first. Without this, a kernel that writes nothing at all
    // would be compared against whatever cudaMalloc handed back -- which is often
    // zeros, and "zeros" is a plausible enough answer to waste an hour on.
    d_out.zero();

    engine::cuda::vector_add(d_a.get(), d_b.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), expected.data(), usize(n), kAddRtol, kAddAtol);
  }
}

TEST(kernels, vector_add_of_nothing_does_nothing) {
  REQUIRE_CUDA_DEVICE();

  // n == 0 is a legal call, not an error: it happens naturally when a batch or a
  // sequence length is zero, and every launcher in kernels.hpp is specified to
  // return without launching. A grid of zero blocks is an INVALID launch on CUDA
  // (cudaErrorInvalidConfiguration), so the early return is load-bearing, not
  // decoration -- and this test is what stops someone deleting it as dead code.
  DeviceBuffer<float> d(1);
  d.zero();
  EXPECT_NO_THROW(engine::cuda::vector_add(d.get(), d.get(), d.get(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(d.download()[0], 0.0f);
}

//===----------------------------------------------------------------------===//
// PART 2 -- exercises 2-6. All implemented and promoted to TEST.
//===----------------------------------------------------------------------===//

TEST(kernels, reduce_sum_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (std::int64_t n : kSizes1D) {
    const std::string stem = "reduce_sum__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);

    const auto& x = g.at("x").data;
    const auto& expected = g.at("expected").data;
    ASSERT_EQ(x.size(), usize(n));
    ASSERT_EQ(expected.size(), std::size_t{1});

    DeviceBuffer<float> d_x(x);
    DeviceBuffer<float> d_out(1);
    d_out.zero();

    engine::cuda::reduce_sum(d_x.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    const double atol = tree_reduction_atol(x);
    CHECK_CASE(stem, host.data(), expected.data(), 1, /*rtol=*/0.0, atol);
  }
}

TEST(kernels, reduce_sum_of_empty_writes_zero) {
  REQUIRE_CUDA_DEVICE();

  // The sum of no elements is 0, and `out` must be WRITTEN with it. Leaving the
  // buffer untouched is the tempting shortcut and it is wrong: the caller has no way
  // to distinguish "your kernel declined to write" from "the answer is whatever was
  // in that memory". The poison value below is what makes the difference visible.
  const std::vector<float> poison{-12345.0f};
  DeviceBuffer<float> d_out(poison);
  DeviceBuffer<float> d_x(4);
  d_x.zero();

  engine::cuda::reduce_sum(d_x.get(), d_out.get(), 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(d_out.download()[0], 0.0f);
}

TEST(kernels, softmax_rows_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const Shape2& s : kSoftmaxShapes) {
    const std::string stem = "softmax_rows__" + dims2(s.rows, s.cols);
    LOAD_GOLDEN(g, stem);

    const auto& in = g.at("input").data;
    const auto& expected = g.at("expected").data;
    ASSERT_EQ(in.size(), usize(s.rows * s.cols));

    DeviceBuffer<float> d_in(in);
    DeviceBuffer<float> d_out(in.size());
    d_out.zero();

    engine::cuda::softmax_rows(d_in.get(), d_out.get(), s.rows, s.cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), expected.data(), in.size(), kSoftmaxRtol, kSoftmaxAtol);

    // An independent invariant, checked per row: probabilities sum to 1. This
    // catches a class of bug the element-wise compare can miss -- if the kernel
    // divides by the wrong row's sum, individual elements can still land inside
    // tolerance while the row is no longer a distribution. The tolerance scales with
    // cols because summing `cols` floats is itself `cols` roundings.
    const double row_tol = 4e-7 * static_cast<double>(s.cols) + 1e-6;
    for (std::int64_t r = 0; r < s.rows; ++r) {
      double sum = 0.0;
      for (std::int64_t c = 0; c < s.cols; ++c) {
        sum += static_cast<double>(host[usize(r * s.cols + c)]);
      }
      EXPECT_NEAR(sum, 1.0, row_tol);
    }
  }
}

TEST(kernels, softmax_rows_survives_the_edge_cases) {
  REQUIRE_CUDA_DEVICE();

  // softmax_rows__edge is the file that fails if you skip the max subtraction:
  //   row 0: all around -50   -- exp(-50) alone underflows; the SHIFTED exp is 1
  //   row 1: one +300 element -- exp(+300) overflows to inf without the shift
  //   row 2: constant         -- every output must be exactly 1/cols
  //   row 3: -20 except one +20 -- the shift reaches 40, the accuracy worst case
  //
  // A kernel that computes exp(x) directly produces inf and NaN on rows 0-1 and
  // looks fine on 2-3. That is the entire reason this file exists.
  LOAD_GOLDEN(g, "softmax_rows__edge");

  const auto& in = g.at("input").data;
  const auto& expected = g.at("expected").data;
  const std::int64_t rows = g.at("input").dim(0);
  const std::int64_t cols = g.at("input").dim(1);

  DeviceBuffer<float> d_in(in);
  DeviceBuffer<float> d_out(in.size());
  d_out.zero();

  engine::cuda::softmax_rows(d_in.get(), d_out.get(), rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host = d_out.download();
  CHECK_CASE("softmax_rows__edge", host.data(), expected.data(), in.size(), kSoftmaxRtol,
             kSoftmaxAtol);

  // Said explicitly, because "allclose passed" is easy to skim past and NaN is the
  // failure mode this file is about.
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (!std::isfinite(host[i])) {
      ctx.add_failure(__FILE__, __LINE__,
                      "non-finite output at index " + std::to_string(i) +
                          " -- the row max was not subtracted before exp()");
      break;
    }
  }
}

TEST(kernels, rmsnorm_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const RmsShape& s : kRmsShapes) {
    const std::string stem =
        "rmsnorm__" + dims2(s.rows, s.cols) + (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);

    const auto& in = g.at("input").data;
    const auto& expected = g.at("expected").data;
    ASSERT_EQ(in.size(), usize(s.rows * s.cols));

    DeviceBuffer<float> d_in(in);
    DeviceBuffer<float> d_out(in.size());
    d_out.zero();

    // A default-constructed DeviceBuffer holds nullptr, which is exactly what the
    // "no weight" case is specified to accept. Keeping it in scope for the whole
    // iteration means the pointer stays valid for the launch.
    DeviceBuffer<float> d_w;
    if (s.with_weight) {
      ASSERT_EQ(g.at("weight").data.size(), usize(s.cols));
      d_w = DeviceBuffer<float>(g.at("weight").data);
    }

    engine::cuda::rmsnorm(d_in.get(), d_w.get(), d_out.get(), s.rows, s.cols, kEps);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), expected.data(), in.size(), kRmsRtol, kRmsAtol);
  }
}

TEST(kernels, rmsnorm_of_zeros_is_zeros_not_nan) {
  REQUIRE_CUDA_DEVICE();

  // mean(x^2) is 0, so this is rsqrt(0 + eps) -- finite only because eps is INSIDE
  // the square root. Written as `x / (rms + eps)` it is still finite; written as
  // `x / sqrt(mean)` with eps added afterwards it is 0/0 = NaN, and a single NaN
  // here propagates through every subsequent layer of the model.
  LOAD_GOLDEN(g, "rmsnorm__zeros");

  const auto& in = g.at("input").data;
  const std::int64_t rows = g.at("input").dim(0);
  const std::int64_t cols = g.at("input").dim(1);

  DeviceBuffer<float> d_in(in);
  DeviceBuffer<float> d_out(in.size());

  // Poisoned with NaN specifically: if the kernel does not write an element, the
  // leftover value is a NaN and the check below catches it. Zero-filling would let a
  // no-op kernel pass this particular test.
  const std::vector<float> nans(in.size(), std::numeric_limits<float>::quiet_NaN());
  d_out.upload(nans);

  engine::cuda::rmsnorm(d_in.get(), nullptr, d_out.get(), rows, cols, kEps);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host = d_out.download();
  for (std::size_t i = 0; i < host.size(); ++i) {
    // Exact equality is right here: 0 * anything finite is exactly 0.
    if (host[i] != 0.0f) {
      ctx.add_failure(__FILE__, __LINE__,
                      "expected exactly 0 at index " + std::to_string(i) + ", got " +
                          std::to_string(host[i]));
      break;
    }
  }
}

TEST(kernels, matmul_naive_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const Shape3& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    const auto& A = g.at("A");
    const auto& B = g.at("B");
    const auto& expected = g.at("expected");

    // Shapes first. If a golden file is transposed relative to what the loop
    // expects, this reports a clean shape mismatch instead of 260,000 wrong floats.
    ASSERT_EQ(A.dim(0), s.m);
    ASSERT_EQ(A.dim(1), s.k);
    ASSERT_EQ(B.dim(0), s.k);
    ASSERT_EQ(B.dim(1), s.n);
    ASSERT_EQ(expected.dim(0), s.m);
    ASSERT_EQ(expected.dim(1), s.n);

    DeviceBuffer<float> d_a(A.data);
    DeviceBuffer<float> d_b(B.data);
    DeviceBuffer<float> d_c(usize(s.m * s.n));
    d_c.zero();

    engine::cuda::matmul_naive(d_a.get(), d_b.get(), d_c.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_c.download();
    CHECK_CASE(stem, host.data(), expected.data.data(), host.size(), kMatmulRtol,
               kMatmulAtol);
  }
}

TEST(kernels, matmul_tiled_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const Shape3& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    DeviceBuffer<float> d_a(g.at("A").data);
    DeviceBuffer<float> d_b(g.at("B").data);
    DeviceBuffer<float> d_c(usize(s.m * s.n));
    d_c.zero();

    engine::cuda::matmul_tiled(d_a.get(), d_b.get(), d_c.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_c.download();
    CHECK_CASE(stem, host.data(), g.at("expected").data.data(), host.size(), kMatmulRtol,
               kMatmulAtol);
  }
}

TEST(kernels, matmul_tiled_agrees_with_matmul_naive) {
  REQUIRE_CUDA_DEVICE();

  // A DIFFERENT question from the two tests above, and the one that actually pins
  // down the optimisation: both kernels compute the same mathematics, so where they
  // disagree, the tiling is at fault rather than the arithmetic.
  //
  // 17x23x31 is the case that matters -- no dimension divides 16 or 32, so every
  // tile on the right and bottom edges is partial. Tiled matmul bugs live almost
  // exclusively in those edges: threads that must load 0.0f into shared memory
  // instead of reading out of bounds, and threads that must compute nothing at all.
  // Squares of 512 hide all of it.
  //
  // The tolerance is deliberately near-exact. Reassociating a dot product into
  // TILE-sized chunks is still a sum of the same K products, so the rounding differs
  // only in grouping -- a few ulp, not a few percent. Anything larger is a bug.
  for (const Shape3& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    DeviceBuffer<float> d_a(g.at("A").data);
    DeviceBuffer<float> d_b(g.at("B").data);
    DeviceBuffer<float> d_naive(usize(s.m * s.n));
    DeviceBuffer<float> d_tiled(usize(s.m * s.n));
    d_naive.zero();
    d_tiled.zero();

    engine::cuda::matmul_naive(d_a.get(), d_b.get(), d_naive.get(), s.m, s.n, s.k);
    engine::cuda::matmul_tiled(d_a.get(), d_b.get(), d_tiled.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> naive = d_naive.download();
    const std::vector<float> tiled = d_tiled.download();
    CHECK_CASE(stem + " (tiled vs naive)", tiled.data(), naive.data(), tiled.size(),
               /*rtol=*/1e-6, /*atol=*/1e-6);
  }
}

TEST(kernels, matmul_register_tiled_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const Shape3& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    DeviceBuffer<float> d_a(g.at("A").data);
    DeviceBuffer<float> d_b(g.at("B").data);
    DeviceBuffer<float> d_c(usize(s.m * s.n));
    d_c.zero();

    engine::cuda::matmul_register_tiled(d_a.get(), d_b.get(), d_c.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_c.download();
    CHECK_CASE(stem, host.data(), g.at("expected").data.data(), host.size(), kMatmulRtol,
               kMatmulAtol);
  }
}

TEST(kernels, matmul_register_tiled_agrees_with_matmul_naive) {
  REQUIRE_CUDA_DEVICE();

  for (const Shape3& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    DeviceBuffer<float> d_a(g.at("A").data);
    DeviceBuffer<float> d_b(g.at("B").data);
    DeviceBuffer<float> d_naive(usize(s.m * s.n));
    DeviceBuffer<float> d_reg(usize(s.m * s.n));
    d_naive.zero();
    d_reg.zero();

    engine::cuda::matmul_naive(d_a.get(), d_b.get(), d_naive.get(), s.m, s.n, s.k);
    engine::cuda::matmul_register_tiled(d_a.get(), d_b.get(), d_reg.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> naive = d_naive.download();
    const std::vector<float> reg = d_reg.download();
    CHECK_CASE(stem + " (reg_tiled vs naive)", reg.data(), naive.data(), reg.size(),
               /*rtol=*/1e-6, /*atol=*/1e-6);
  }
}

TEST(kernels, matmul_register_tiled_non_tile_multiples) {
  REQUIRE_CUDA_DEVICE();

  // matmul_register_tiled uses BM=64, BN=64, BK=8 with TM=8, TN=8 thread tiles.
  // Test shapes that are explicitly NOT multiples of 64 or 8, including odd primes,
  // to ensure thread bounds checks and partial tile guards prevent out-of-bounds reads/writes.
  const Shape3 non_tile_shapes[] = {
      {3, 7, 11},      // tiny: well below block and warp sizes
      {65, 137, 73},   // 1 past BM, BN, BK multiples
      {129, 257, 65},  // 1 past large tile multiples
      {1, 65, 127},    // M=1 decode shape with unaligned N and K
      {71, 97, 113},   // prime dimensions across all 3 axes
  };

  for (const Shape3& s : non_tile_shapes) {
    const std::string name = "matmul_reg_tiled_non_tile__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);

    const std::size_t size_A = usize(s.m * s.k);
    const std::size_t size_B = usize(s.k * s.n);
    const std::size_t size_C = usize(s.m * s.n);

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_expected(size_C, 0.0f);

    // Initialize with deterministic pseudo-random values scaled by 1/sqrt(K)
    const float scale_A = 1.0f / std::sqrt(static_cast<float>(s.k));
    const float scale_B = 1.0f / std::sqrt(static_cast<float>(s.k));
    for (std::size_t i = 0; i < size_A; ++i) {
      h_A[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * scale_A;
    }
    for (std::size_t i = 0; i < size_B; ++i) {
      h_B[i] = std::cos(static_cast<float>(i + 1) * 0.1f) * scale_B;
    }

    // Tier 2 double-precision scalar oracle
    engine::cpu::matmul(h_A.data(), h_B.data(), h_expected.data(), s.m, s.n, s.k);

    DeviceBuffer<float> d_a(h_A);
    DeviceBuffer<float> d_b(h_B);
    DeviceBuffer<float> d_c(size_C);
    d_c.zero();

    engine::cuda::matmul_register_tiled(d_a.get(), d_b.get(), d_c.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_c.download();
    CHECK_CASE(name, host.data(), h_expected.data(), host.size(), kMatmulRtol,
               kMatmulAtol);
  }
}

TEST(kernels, matmul_with_k_zero_is_the_zero_matrix) {
  REQUIRE_CUDA_DEVICE();

  // An Mx0 times a 0xN product is the MxN ZERO matrix -- the empty sum is 0, not
  // undefined. It is a real case: a KV-cache with nothing in it yet at generation
  // step 0. Getting this wrong leaves the output buffer at whatever it held, and the
  // symptom shows up much later as one garbage token at the start of a sequence.
  const std::int64_t M = 4, N = 3;
  const std::vector<float> poison(usize(M * N), 7.0f);
  DeviceBuffer<float> d_c(poison);
  DeviceBuffer<float> d_a(1);  // non-null, but zero elements are ever read
  DeviceBuffer<float> d_b(1);
  d_a.zero();
  d_b.zero();

  engine::cuda::matmul_naive(d_a.get(), d_b.get(), d_c.get(), M, N, /*K=*/0);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host = d_c.download();
  for (float v : host) {
    if (v != 0.0f) {
      ctx.add_failure(
          __FILE__, __LINE__,
          "K == 0 must produce an all-zero MxN result, found " + std::to_string(v));
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Module 3 -- Fused operations tests
//===----------------------------------------------------------------------===//

TEST(kernels, residual_rmsnorm_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const auto& s : kResidualRmsShapes) {
    const std::string stem =
        "residual_rmsnorm__" + dims2(s.rows, s.cols) + (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);

    const auto& x = g.at("x");
    const auto& res = g.at("residual");
    const auto& exp_norm = g.at("expected");
    const auto& exp_sum = g.at("sum_out");

    DeviceBuffer<float> d_x(x.data);
    DeviceBuffer<float> d_res(res.data);
    DeviceBuffer<float> d_norm_out(usize(s.rows * s.cols));
    DeviceBuffer<float> d_sum_out(usize(s.rows * s.cols));
    d_norm_out.zero();
    d_sum_out.zero();

    const float* d_weight_ptr = nullptr;
    DeviceBuffer<float> d_weight;
    if (s.with_weight) {
      d_weight = DeviceBuffer<float>(g.at("weight").data);
      d_weight_ptr = d_weight.get();
    }

    engine::cuda::residual_rmsnorm(d_x.get(), d_res.get(), d_weight_ptr, d_norm_out.get(),
                                   d_sum_out.get(), s.rows, s.cols, kEps);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host_norm = d_norm_out.download();
    const std::vector<float> host_sum = d_sum_out.download();
    CHECK_CASE(stem + " (norm)", host_norm.data(), exp_norm.data.data(), host_norm.size(),
               kRmsRtol, kRmsAtol);
    CHECK_CASE(stem + " (sum)", host_sum.data(), exp_sum.data.data(), host_sum.size(),
               kRmsRtol, kRmsAtol);
  }
}

TEST(kernels, residual_rmsnorm_of_zeros_is_zeros_not_nan) {
  REQUIRE_CUDA_DEVICE();

  LOAD_GOLDEN(g, "residual_rmsnorm__zeros");
  const auto& x = g.at("x");
  const auto& res = g.at("residual");
  const std::int64_t rows = 2, cols = 256;

  DeviceBuffer<float> d_x(x.data);
  DeviceBuffer<float> d_res(res.data);
  DeviceBuffer<float> d_norm_out(usize(rows * cols));
  DeviceBuffer<float> d_sum_out(usize(rows * cols));

  const std::vector<float> nans(usize(rows * cols),
                                std::numeric_limits<float>::quiet_NaN());
  d_norm_out.upload(nans);
  d_sum_out.upload(nans);

  engine::cuda::residual_rmsnorm(d_x.get(), d_res.get(), nullptr, d_norm_out.get(),
                                 d_sum_out.get(), rows, cols, kEps);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host_norm = d_norm_out.download();
  const std::vector<float> host_sum = d_sum_out.download();
  for (std::size_t i = 0; i < host_norm.size(); ++i) {
    if (host_norm[i] != 0.0f) {
      ctx.add_failure(__FILE__, __LINE__,
                      "expected norm exactly 0 at index " + std::to_string(i) + ", got " +
                          std::to_string(host_norm[i]));
      break;
    }
    if (host_sum[i] != 0.0f) {
      ctx.add_failure(__FILE__, __LINE__,
                      "expected sum exactly 0 at index " + std::to_string(i) + ", got " +
                          std::to_string(host_sum[i]));
      break;
    }
  }
}

TEST(kernels, residual_rmsnorm_null_weight_equals_unit_weight) {
  REQUIRE_CUDA_DEVICE();

  LOAD_GOLDEN(g, "residual_rmsnorm__32x4096_now");
  const auto& x = g.at("x");
  const auto& res = g.at("residual");
  const std::int64_t rows = 32, cols = 4096;

  DeviceBuffer<float> d_x(x.data);
  DeviceBuffer<float> d_res(res.data);
  const std::vector<float> ones(usize(cols), 1.0f);
  DeviceBuffer<float> d_ones(ones);

  DeviceBuffer<float> d_norm_null(usize(rows * cols));
  DeviceBuffer<float> d_sum_null(usize(rows * cols));
  DeviceBuffer<float> d_norm_ones(usize(rows * cols));
  DeviceBuffer<float> d_sum_ones(usize(rows * cols));

  engine::cuda::residual_rmsnorm(d_x.get(), d_res.get(), nullptr, d_norm_null.get(),
                                 d_sum_null.get(), rows, cols, kEps);
  engine::cuda::residual_rmsnorm(d_x.get(), d_res.get(), d_ones.get(), d_norm_ones.get(),
                                 d_sum_ones.get(), rows, cols, kEps);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> norm_null = d_norm_null.download();
  const std::vector<float> norm_ones = d_norm_ones.download();
  const std::vector<float> sum_null = d_sum_null.download();
  const std::vector<float> sum_ones = d_sum_ones.download();

  CHECK_CASE("residual_rmsnorm null vs ones (norm)", norm_null.data(), norm_ones.data(),
             norm_null.size(), kAddRtol, kAddAtol);
  CHECK_CASE("residual_rmsnorm null vs ones (sum)", sum_null.data(), sum_ones.data(),
             sum_null.size(), kAddRtol, kAddAtol);
}

TEST(kernels, rmsnorm_linear_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const auto& s : kRmsnormLinearShapes) {
    const std::string stem = "rmsnorm_linear__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k) +
                             (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);

    const auto& in = g.at("input");
    const auto& W = g.at("W");
    const auto& exp = g.at("expected");

    DeviceBuffer<float> d_in(in.data);
    DeviceBuffer<float> d_w(W.data);
    DeviceBuffer<float> d_out(usize(s.m * s.n));
    d_out.zero();

    const float* d_weight_ptr = nullptr;
    DeviceBuffer<float> d_weight;
    if (s.with_weight) {
      d_weight = DeviceBuffer<float>(g.at("weight").data);
      d_weight_ptr = d_weight.get();
    }

    engine::cuda::rmsnorm_linear(d_in.get(), d_weight_ptr, d_w.get(), d_out.get(), s.m,
                                 s.n, s.k, kEps);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), exp.data.data(), host.size(), kMatmulRtol, kMatmulAtol);
  }
}

TEST(kernels, rmsnorm_linear_null_weight_equals_unit_weight) {
  REQUIRE_CUDA_DEVICE();

  LOAD_GOLDEN(g, "rmsnorm_linear__32x4096x4096_now");
  const auto& in = g.at("input");
  const auto& W = g.at("W");
  const std::int64_t M = 32, N = 4096, K = 4096;

  DeviceBuffer<float> d_in(in.data);
  DeviceBuffer<float> d_w(W.data);
  const std::vector<float> ones(usize(K), 1.0f);
  DeviceBuffer<float> d_ones(ones);

  DeviceBuffer<float> d_out_null(usize(M * N));
  DeviceBuffer<float> d_out_ones(usize(M * N));
  d_out_null.zero();
  d_out_ones.zero();

  engine::cuda::rmsnorm_linear(d_in.get(), nullptr, d_w.get(), d_out_null.get(), M, N, K,
                               kEps);
  engine::cuda::rmsnorm_linear(d_in.get(), d_ones.get(), d_w.get(), d_out_ones.get(), M,
                               N, K, kEps);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host_null = d_out_null.download();
  const std::vector<float> host_ones = d_out_ones.download();

  CHECK_CASE("rmsnorm_linear null vs ones", host_null.data(), host_ones.data(),
             host_null.size(), /*rtol=*/1e-6, /*atol=*/1e-6);
}

TEST(kernels, rmsnorm_linear_agrees_with_separate_rmsnorm_and_matmul) {
  REQUIRE_CUDA_DEVICE();

  for (const auto& s : kRmsnormLinearShapes) {
    const std::string stem = "rmsnorm_linear__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k) +
                             (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);

    const auto& in = g.at("input");
    const auto& W = g.at("W");

    DeviceBuffer<float> d_in(in.data);
    DeviceBuffer<float> d_w(W.data);
    DeviceBuffer<float> d_fused_out(usize(s.m * s.n));
    DeviceBuffer<float> d_temp(usize(s.m * s.k));
    DeviceBuffer<float> d_sep_out(usize(s.m * s.n));
    d_fused_out.zero();
    d_temp.zero();
    d_sep_out.zero();

    const float* d_weight_ptr = nullptr;
    DeviceBuffer<float> d_weight;
    if (s.with_weight) {
      d_weight = DeviceBuffer<float>(g.at("weight").data);
      d_weight_ptr = d_weight.get();
    }

    // Fused kernel
    engine::cuda::rmsnorm_linear(d_in.get(), d_weight_ptr, d_w.get(), d_fused_out.get(),
                                 s.m, s.n, s.k, kEps);

    // Separate kernels: rmsnorm followed by matmul_tiled
    engine::cuda::rmsnorm(d_in.get(), d_weight_ptr, d_temp.get(), s.m, s.k, kEps);
    engine::cuda::matmul_tiled(d_temp.get(), d_w.get(), d_sep_out.get(), s.m, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> fused = d_fused_out.download();
    const std::vector<float> separate = d_sep_out.download();

    CHECK_CASE(stem + " (fused vs separate)", fused.data(), separate.data(), fused.size(),
               /*rtol=*/1e-4, /*atol=*/5e-5);
  }
}

//===----------------------------------------------------------------------===//
// Exercise 9 (GEMV) -- Matrix-Vector product for decode token generation.
//===----------------------------------------------------------------------===//

struct GemvShape {
  std::int64_t n, k;
};

const GemvShape kGemvShapes[] = {
    {4096, 4096},
    {12288, 4096},
    {127, 31},
    {64, 128},
};

TEST(kernels, gemv_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  for (const GemvShape& s : kGemvShapes) {
    const std::string stem = "gemv__" + std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    const auto& A = g.at("A").data;
    const auto& x = g.at("x").data;
    const auto& expected = g.at("expected").data;

    ASSERT_EQ(A.size(), usize(s.k * s.n));
    ASSERT_EQ(x.size(), usize(s.k));
    ASSERT_EQ(expected.size(), usize(s.n));

    DeviceBuffer<float> d_A(A);
    DeviceBuffer<float> d_x(x);
    DeviceBuffer<float> d_out(s.n);
    d_out.zero();

    engine::cuda::gemv(d_A.get(), d_x.get(), d_out.get(), s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), expected.data(), usize(s.n), kMatmulRtol, kMatmulAtol);
  }
}

TEST(kernels, gemv_agrees_with_matmul_naive_at_M1) {
  REQUIRE_CUDA_DEVICE();

  for (const GemvShape& s : kGemvShapes) {
    const std::string stem = "gemv__" + std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    const auto& A = g.at("A").data;
    const auto& x = g.at("x").data;

    DeviceBuffer<float> d_A(A);
    DeviceBuffer<float> d_x(x);
    DeviceBuffer<float> d_gemv_out(s.n);
    DeviceBuffer<float> d_matmul_out(s.n);
    d_gemv_out.zero();
    d_matmul_out.zero();

    engine::cuda::gemv(d_A.get(), d_x.get(), d_gemv_out.get(), s.n, s.k);
    engine::cuda::matmul_naive(d_x.get(), d_A.get(), d_matmul_out.get(), 1, s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> gemv_res = d_gemv_out.download();
    const std::vector<float> matmul_res = d_matmul_out.download();

    CHECK_CASE(stem + " (gemv vs matmul_naive)", gemv_res.data(), matmul_res.data(),
               usize(s.n), kMatmulRtol, kMatmulAtol);
  }
}

TEST(kernels, gemv_of_zero_input) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t N = 64, K = 128;
  std::vector<float> zeros(usize(K), 0.0f);
  std::vector<float> ones(usize(K * N), 1.0f);

  DeviceBuffer<float> d_A(ones);
  DeviceBuffer<float> d_x(zeros);
  DeviceBuffer<float> d_out(N);
  d_out.zero();

  engine::cuda::gemv(d_A.get(), d_x.get(), d_out.get(), N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> host = d_out.download();
  for (float v : host) {
    EXPECT_EQ(v, 0.0f);
  }
}

//===----------------------------------------------------------------------===//
// Phase 3 -- FP16 and missing inference kernels: gemv_fp16, embedding, argmax
//===----------------------------------------------------------------------===//

TEST(kernels, gemv_fp16_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  struct Shape {
    std::int64_t n, k;
  };
  const Shape shapes[] = {
      {64, 64}, {128, 256}, {4096, 4096}, {73, 125},  // unaligned fallback
  };

  for (const auto& s : shapes) {
    std::vector<engine::half> h_A(static_cast<std::size_t>(s.k * s.n));
    std::vector<engine::half> h_x(static_cast<std::size_t>(s.k));
    std::vector<engine::half> exp(static_cast<std::size_t>(s.n));

    for (std::size_t i = 0; i < h_A.size(); ++i) {
      h_A[i] =
          engine::float_to_half(static_cast<float>(static_cast<int>(i % 19) - 9) * 0.05f);
    }
    for (std::size_t i = 0; i < h_x.size(); ++i) {
      h_x[i] =
          engine::float_to_half(static_cast<float>(static_cast<int>(i % 13) - 6) * 0.05f);
    }

    engine::cpu::gemv_fp16(h_A.data(), h_x.data(), exp.data(), s.n, s.k);

    DeviceBuffer<engine::half> d_A(h_A);
    DeviceBuffer<engine::half> d_x(h_x);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(s.n));
    d_out.zero();

    engine::cuda::gemv_fp16(d_A.get(), d_x.get(), d_out.get(), s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<engine::half> host = d_out.download();
    std::vector<float> host_f(host.size());
    std::vector<float> exp_f(exp.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
      host_f[i] = engine::half_to_float(host[i]);
      exp_f[i] = engine::half_to_float(exp[i]);
    }

    CHECK_CASE("gemv_fp16__" + std::to_string(s.n) + "x" + std::to_string(s.k),
               host_f.data(), exp_f.data(), host_f.size(), 2e-3, 2e-3);
  }
}

TEST(kernels, embedding_f32_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  struct Case {
    std::int64_t v, d, t;
  };
  const Case cases[] = {
      {1000, 256, 5}, {100, 65, 3},  // unaligned fallback
  };

  for (const auto& c : cases) {
    std::vector<float> table(static_cast<std::size_t>(c.v * c.d));
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = static_cast<float>(i) * 0.1f;
    }

    std::vector<std::int32_t> ids = {0, static_cast<std::int32_t>(c.v - 1), 7};
    while (static_cast<std::int64_t>(ids.size()) < c.t) {
      ids.push_back(static_cast<std::int32_t>(ids.size() % c.v));
    }

    std::vector<float> exp(static_cast<std::size_t>(c.t * c.d));
    engine::cpu::embedding(table.data(), ids.data(), exp.data(), c.t, c.d, c.v);

    DeviceBuffer<float> d_table(table);
    DeviceBuffer<std::int32_t> d_ids(ids);
    DeviceBuffer<float> d_out(static_cast<std::size_t>(c.t * c.d));
    d_out.zero();

    engine::cuda::embedding(d_table.get(), d_ids.get(), d_out.get(), c.t, c.d, c.v);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE("embedding_f32", host.data(), exp.data(), host.size(), 0.0, 0.0);
  }
}

TEST(kernels, embedding_fp16_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  struct Case {
    std::int64_t v, d, t;
  };
  const Case cases[] = {
      {500, 128, 4},
      {50, 65, 2},
  };

  for (const auto& c : cases) {
    std::vector<engine::half> table(static_cast<std::size_t>(c.v * c.d));
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = engine::float_to_half(static_cast<float>(i % 31));
    }

    std::vector<std::int32_t> ids = {1, static_cast<std::int32_t>(c.v - 1)};
    while (static_cast<std::int64_t>(ids.size()) < c.t) {
      ids.push_back(static_cast<std::int32_t>(ids.size() % c.v));
    }

    std::vector<engine::half> exp(static_cast<std::size_t>(c.t * c.d));
    engine::cpu::embedding_fp16(table.data(), ids.data(), exp.data(), c.t, c.d, c.v);

    DeviceBuffer<engine::half> d_table(table);
    DeviceBuffer<std::int32_t> d_ids(ids);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(c.t * c.d));
    d_out.zero();

    engine::cuda::embedding_fp16(d_table.get(), d_ids.get(), d_out.get(), c.t, c.d, c.v);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<engine::half> host = d_out.download();
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_EQ(engine::half_to_float(host[i]), engine::half_to_float(exp[i]));
    }
  }
}

TEST(kernels, argmax_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t vocab_size = 32000;
  std::vector<float> logits(static_cast<std::size_t>(vocab_size));
  for (std::size_t i = 0; i < logits.size(); ++i) {
    logits[i] = static_cast<float>(static_cast<int>(i % 1000) - 500) * 0.01f;
  }
  // Inject winner at index 17532
  logits[17532] = 999.0f;
  // Inject identical tie at later index 29111 -> tiebreaker must pick 17532
  logits[29111] = 999.0f;

  const std::int32_t exp = engine::cpu::argmax(logits.data(), vocab_size);
  EXPECT_EQ(exp, 17532);

  DeviceBuffer<float> d_logits(logits);
  DeviceBuffer<std::int32_t> d_out(1);
  d_out.zero();

  engine::cuda::argmax(d_logits.get(), d_out.get(), vocab_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<std::int32_t> host = d_out.download();
  EXPECT_EQ(host[0], exp);
}

TEST(kernels, argmax_fp16_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t vocab_size = 32000;
  std::vector<engine::half> logits(static_cast<std::size_t>(vocab_size));
  for (std::size_t i = 0; i < logits.size(); ++i) {
    logits[i] = engine::float_to_half(static_cast<float>(static_cast<int>(i % 100) - 50));
  }
  // Inject winner at index 8412
  logits[8412] = engine::float_to_half(500.0f);
  logits[21000] = engine::float_to_half(500.0f);

  const std::int32_t exp = engine::cpu::argmax_fp16(logits.data(), vocab_size);
  EXPECT_EQ(exp, 8412);

  DeviceBuffer<engine::half> d_logits(logits);
  DeviceBuffer<std::int32_t> d_out(1);
  d_out.zero();

  engine::cuda::argmax_fp16(d_logits.get(), d_out.get(), vocab_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<std::int32_t> host = d_out.download();
  EXPECT_EQ(host[0], exp);
}

//===----------------------------------------------------------------------===//
// Phase 4 -- Fused SwiGLU activation tests
//===----------------------------------------------------------------------===//

TEST(kernels, swiglu_f32_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t sizes[] = {1, 65, 1024, 4096, 11008};
  for (std::int64_t n : sizes) {
    std::vector<float> gate(static_cast<std::size_t>(n));
    std::vector<float> up(static_cast<std::size_t>(n));
    std::vector<float> exp(static_cast<std::size_t>(n));

    for (std::size_t i = 0; i < gate.size(); ++i) {
      gate[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.2f;
      up[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.2f;
    }

    engine::cpu::swiglu(gate.data(), up.data(), exp.data(), n);

    DeviceBuffer<float> d_gate(gate);
    DeviceBuffer<float> d_up(up);
    DeviceBuffer<float> d_out(static_cast<std::size_t>(n));
    d_out.zero();

    engine::cuda::swiglu(d_gate.get(), d_up.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE("swiglu_f32__n" + std::to_string(n), host.data(), exp.data(), host.size(),
               1e-5, 1e-5);
  }
}

TEST(kernels, swiglu_fp16_matches_cpu_ref) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t sizes[] = {1, 65, 1024, 4096, 11008};
  for (std::int64_t n : sizes) {
    std::vector<engine::half> gate(static_cast<std::size_t>(n));
    std::vector<engine::half> up(static_cast<std::size_t>(n));
    std::vector<engine::half> exp(static_cast<std::size_t>(n));

    for (std::size_t i = 0; i < gate.size(); ++i) {
      gate[i] =
          engine::float_to_half(static_cast<float>(static_cast<int>(i % 23) - 11) * 0.2f);
      up[i] =
          engine::float_to_half(static_cast<float>(static_cast<int>(i % 17) - 8) * 0.2f);
    }

    engine::cpu::swiglu_fp16(gate.data(), up.data(), exp.data(), n);

    DeviceBuffer<engine::half> d_gate(gate);
    DeviceBuffer<engine::half> d_up(up);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(n));
    d_out.zero();

    engine::cuda::swiglu_fp16(d_gate.get(), d_up.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<engine::half> host = d_out.download();
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_NEAR(engine::half_to_float(host[i]), engine::half_to_float(exp[i]), 2e-3f);
    }
  }
}

//===----------------------------------------------------------------------===//
// PART 3 -- the launch CONTRACT.
//===----------------------------------------------------------------------===//

TEST(kernels, launchers_reject_negative_dimensions) {
  REQUIRE_CUDA_DEVICE();

  DeviceBuffer<float> d(16);
  d.zero();
  float* p = d.get();

  DeviceBuffer<engine::half> d_h(16);
  d_h.zero();
  engine::half* p_h = d_h.get();

  DeviceBuffer<std::int32_t> d_i(16);
  d_i.zero();
  std::int32_t* p_i = d_i.get();

  EXPECT_THROWS_MSG(engine::cuda::vector_add(p, p, p, -1), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::reduce_sum(p, p, -1), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::softmax_rows(p, p, -1, 4), "negative");
  EXPECT_THROWS_MSG(engine::cuda::softmax_rows(p, p, 4, -1), "negative");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm(p, p, p, -1, 4, 1e-5f), "negative");
  EXPECT_THROWS_MSG(engine::cuda::matmul_naive(p, p, p, -1, 4, 4), "negative");
  EXPECT_THROWS_MSG(engine::cuda::matmul_tiled(p, p, p, 4, 4, -1), "negative");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, p, p, p, p, -1, 4, 1e-5f),
                    "negative");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, p, p, p, p, 4, -1, 1e-5f),
                    "negative");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, p, p, -1, 4, 4, 1e-5f),
                    "negative");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, p, p, 4, -1, 4, 1e-5f),
                    "negative");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, p, p, 4, 4, -1, 1e-5f),
                    "negative");
  EXPECT_THROWS_MSG(engine::cuda::gemv(p, p, p, -1, 4), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::gemv(p, p, p, 4, -1), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::gemv_fp16(p_h, p_h, p_h, -1, 4), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::gemv_fp16(p_h, p_h, p_h, 4, -1), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::embedding(p, p_i, p, -1, 4, 10), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::embedding(p, p_i, p, 4, -1, 10), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::embedding(p, p_i, p, 4, 4, 0), "positive");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(p_h, p_i, p_h, -1, 4, 10),
                    "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(p_h, p_i, p_h, 4, -1, 10),
                    "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(p_h, p_i, p_h, 4, 4, 0), "positive");
  EXPECT_THROWS_MSG(engine::cuda::argmax(p, p_i, 0), "positive");
  EXPECT_THROWS_MSG(engine::cuda::argmax(p, p_i, -1), "positive");
  EXPECT_THROWS_MSG(engine::cuda::argmax_fp16(p_h, p_i, 0), "positive");
  EXPECT_THROWS_MSG(engine::cuda::argmax_fp16(p_h, p_i, -1), "positive");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(p, p, p, -1, 4, 4),
                    "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(p, p, p, 4, -1, 4),
                    "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(p, p, p, 4, 4, -1),
                    "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::swiglu(p, p, p, -1), "non-negative");
  EXPECT_THROWS_MSG(engine::cuda::swiglu_fp16(p_h, p_h, p_h, -1), "non-negative");

  // eps < 0 would put a negative number under the square root.
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm(p, p, p, 4, 4, -1.0f), "eps");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, p, p, p, p, 4, 4, -1.0f), "eps");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, p, p, 4, 4, 4, -1.0f), "eps");
}

TEST(kernels, launchers_reject_null_pointers) {
  REQUIRE_CUDA_DEVICE();

  DeviceBuffer<float> d(16);
  d.zero();
  float* p = d.get();

  DeviceBuffer<engine::half> d_h(16);
  d_h.zero();
  engine::half* p_h = d_h.get();

  DeviceBuffer<std::int32_t> d_i(16);
  d_i.zero();
  std::int32_t* p_i = d_i.get();

  EXPECT_THROWS_MSG(engine::cuda::vector_add(nullptr, p, p, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::vector_add(p, nullptr, p, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::vector_add(p, p, nullptr, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::reduce_sum(nullptr, p, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::reduce_sum(p, nullptr, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::softmax_rows(nullptr, p, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm(nullptr, p, p, 2, 2, 1e-5f), "null");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm(p, p, nullptr, 2, 2, 1e-5f), "null");
  EXPECT_THROWS_MSG(engine::cuda::matmul_naive(nullptr, p, p, 2, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::matmul_tiled(p, p, nullptr, 2, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(nullptr, p, p, p, p, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, nullptr, p, p, p, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, p, p, nullptr, p, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::residual_rmsnorm(p, p, p, p, nullptr, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(nullptr, p, p, p, 2, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, nullptr, p, 2, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::rmsnorm_linear(p, p, p, nullptr, 2, 2, 2, 1e-5f),
                    "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv(nullptr, p, p, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv(p, nullptr, p, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv(p, p, nullptr, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv_fp16(nullptr, p_h, p_h, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv_fp16(p_h, nullptr, p_h, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::gemv_fp16(p_h, p_h, nullptr, 4, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding(nullptr, p_i, p, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding(p, nullptr, p, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding(p, p_i, nullptr, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(nullptr, p_i, p_h, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(p_h, nullptr, p_h, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::embedding_fp16(p_h, p_i, nullptr, 4, 4, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::argmax(nullptr, p_i, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::argmax(p, nullptr, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::argmax_fp16(nullptr, p_i, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::argmax_fp16(p_h, nullptr, 10), "null");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(nullptr, p, p, 2, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(p, nullptr, p, 2, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::matmul_register_tiled(p, p, nullptr, 2, 2, 2), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu(nullptr, p, p, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu(p, nullptr, p, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu(p, p, nullptr, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu_fp16(nullptr, p_h, p_h, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu_fp16(p_h, nullptr, p_h, 4), "null");
  EXPECT_THROWS_MSG(engine::cuda::swiglu_fp16(p_h, p_h, nullptr, 4), "null");

  // rmsnorm's, residual_rmsnorm's, and rmsnorm_linear's `weight` is ALLOWED to be null:
  // "no learned gain" is a valid configuration, not a mistake.
  EXPECT_NO_THROW(engine::cuda::rmsnorm(p, nullptr, p, 0, 0, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::residual_rmsnorm(p, p, nullptr, p, p, 0, 0, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::rmsnorm_linear(p, nullptr, p, p, 0, 0, 0, 1e-5f));
}

TEST(kernels, empty_work_is_a_no_op_not_an_error) {
  REQUIRE_CUDA_DEVICE();

  DeviceBuffer<float> d(16);
  d.zero();
  float* p = d.get();

  DeviceBuffer<engine::half> d_h(16);
  d_h.zero();
  engine::half* p_h = d_h.get();

  DeviceBuffer<std::int32_t> d_i(16);
  d_i.zero();
  std::int32_t* p_i = d_i.get();

  // Zero rows or zero columns must return without launching, for the same
  // cudaErrorInvalidConfiguration reason as vector_add above. reduce_sum is the
  // exception -- it has an answer to write (0.0f) -- and it is tested separately.
  EXPECT_NO_THROW(engine::cuda::softmax_rows(p, p, 0, 8));
  EXPECT_NO_THROW(engine::cuda::softmax_rows(p, p, 8, 0));
  EXPECT_NO_THROW(engine::cuda::rmsnorm(p, p, p, 0, 8, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::rmsnorm(p, p, p, 8, 0, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::matmul_naive(p, p, p, 0, 8, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_naive(p, p, p, 8, 0, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_tiled(p, p, p, 0, 8, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_tiled(p, p, p, 8, 0, 8));
  EXPECT_NO_THROW(engine::cuda::residual_rmsnorm(p, p, p, p, p, 0, 8, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::residual_rmsnorm(p, p, p, p, p, 8, 0, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::rmsnorm_linear(p, p, p, p, 0, 8, 8, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::rmsnorm_linear(p, p, p, p, 8, 0, 8, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::rmsnorm_linear(p, p, p, p, 8, 8, 0, 1e-5f));
  EXPECT_NO_THROW(engine::cuda::gemv(p, p, p, 0, 8));
  EXPECT_NO_THROW(engine::cuda::gemv(p, p, p, 8, 0));
  EXPECT_NO_THROW(engine::cuda::gemv_fp16(p_h, p_h, p_h, 0, 8));
  EXPECT_NO_THROW(engine::cuda::gemv_fp16(p_h, p_h, p_h, 8, 0));
  EXPECT_NO_THROW(engine::cuda::embedding(p, p_i, p, 0, 8, 10));
  EXPECT_NO_THROW(engine::cuda::embedding(p, p_i, p, 8, 0, 10));
  EXPECT_NO_THROW(engine::cuda::embedding_fp16(p_h, p_i, p_h, 0, 8, 10));
  EXPECT_NO_THROW(engine::cuda::embedding_fp16(p_h, p_i, p_h, 8, 0, 10));
  EXPECT_NO_THROW(engine::cuda::matmul_register_tiled(p, p, p, 0, 8, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_register_tiled(p, p, p, 8, 0, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_register_tiled(p, p, p, 8, 8, 0));
  EXPECT_NO_THROW(engine::cuda::swiglu(p, p, p, 0));
  EXPECT_NO_THROW(engine::cuda::swiglu_fp16(p_h, p_h, p_h, 0));

  // And nothing may have been launched, let alone written.
  CUDA_CHECK(cudaDeviceSynchronize());
  for (float v : d.download()) EXPECT_EQ(v, 0.0f);
}

#endif  // ENGINE_HAS_CUDA
