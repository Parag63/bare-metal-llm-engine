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
#include <engine/cuda_stream.hpp>
#include <engine/device_buffer.hpp>
#include <engine/kernels.hpp>

#include <algorithm>
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
const RmsnormLinearShape kRmsnormLinearShapes[] = {{1, 256, 4096, true},
                                                   {32, 256, 512, true},
                                                   {1, 768, 256, true},
                                                   {17, 127, 31, true},
                                                   {32, 256, 512, false}};

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

std::vector<engine::half> to_half_vec(const std::vector<float>& src) {
  std::vector<engine::half> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    out[i] = engine::float_to_half(src[i]);
  }
  return out;
}

std::vector<float> to_float_vec(const std::vector<engine::half>& src) {
  std::vector<float> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    out[i] = engine::half_to_float(src[i]);
  }
  return out;
}

std::vector<std::int32_t> to_i32_vec(const std::vector<float>& src) {
  std::vector<std::int32_t> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    out[i] = static_cast<std::int32_t>(std::lround(src[i]));
  }
  return out;
}

// Derived tolerance for FP16 attention (same derivation as test_cpu_ref.cpp):
// 2 * eps16 relative (binary16 output round + tie slack) and
// 1.5 * eps16 + 4 * s_kv * u32 absolute.
double attention_rtol() { return 2.0 * std::ldexp(1.0, -10); }
double attention_atol(std::int64_t s_kv) {
  return 1.5 * std::ldexp(1.0, -10) +
         4.0 * static_cast<double>(s_kv) * std::ldexp(1.0, -23);
}
double lse_rtol() { return 2e-5; }
double lse_atol(std::int64_t s_kv) {
  return 2e-5 + 4.0 * static_cast<double>(s_kv) * std::ldexp(1.0, -23);
}

struct AttentionCase {
  const char* name;
  std::int64_t s_q, s_kv, n_heads, n_kv_heads, head_dim, q_pos_offset;
  bool is_causal;
};

const AttentionCase kAttentionCases[] = {
    {"s1_h1_d64_causal", 1, 1, 1, 1, 64, 0, true},
    {"s1_kv31_h4_d64_nocausal", 1, 31, 4, 4, 64, 0, false},
    {"s17_h8_kv2_d64_causal", 17, 17, 8, 2, 64, 0, true},
    {"s33_kv65_h8_kv2_d128_off32_causal", 33, 65, 8, 2, 128, 32, true},
    {"s64_h8_d64_causal", 64, 64, 8, 8, 64, 0, true},
    {"s128_h16_kv4_d128_causal", 128, 128, 16, 4, 128, 0, true},
    {"s16_h32_kv4_d64_causal", 16, 16, 32, 4, 64, 0, true},
    {"s1_kv257_h32_kv4_d64_decode", 1, 257, 32, 4, 64, 256, false},
    {"s1_kv513_h8_kv2_d128_decode", 1, 513, 8, 2, 128, 512, false},
    {"s16_kv32_h4_d64_stress_causal", 16, 32, 4, 4, 64, 0, true},
};

struct RopeCase {
  const char* name;
  std::int64_t s, n_heads, n_kv_heads, head_dim, pos_offset;
  engine::RopeConvention convention;
};

const RopeCase kRopeCases[] = {
    {"s8_h4_kv2_d64_pos0_half", 8, 4, 2, 64, 0, engine::RopeConvention::RotateHalf},
    {"s8_h4_kv2_d64_pos0_adj", 8, 4, 2, 64, 0, engine::RopeConvention::AdjacentPair},
    {"s17_h8_kv2_d128_pos137_half", 17, 8, 2, 128, 137,
     engine::RopeConvention::RotateHalf},
    {"s4_h4_d64_pos2047_half", 4, 4, 4, 64, 2047, engine::RopeConvention::RotateHalf},
    {"s4_h4_d64_pos2047_adj", 4, 4, 4, 64, 2047, engine::RopeConvention::AdjacentPair},
    {"s4_h4_kv2_d128_pos99998_half", 4, 4, 2, 128, 99998,
     engine::RopeConvention::RotateHalf},
    {"s4_h4_kv2_d128_pos99998_adj", 4, 4, 2, 128, 99998,
     engine::RopeConvention::AdjacentPair},
};

using PrefillAttentionFn = void (*)(const engine::half*, const engine::half*,
                                    const engine::half*, engine::half*, std::int64_t,
                                    std::int64_t, std::int64_t, std::int64_t,
                                    std::int64_t, std::int64_t, float, std::int64_t, bool,
                                    float*, cudaStream_t);

void verify_prefill_attention_against_golden(engtest::TestContext& ctx,
                                             PrefillAttentionFn fn, const char* label) {
  for (const auto& c : kAttentionCases) {
    const std::string stem = std::string("attention__") + c.name;
    LOAD_GOLDEN(g, stem);

    const auto q = to_half_vec(g.at("q").data);
    const auto k = to_half_vec(g.at("k").data);
    const auto v = to_half_vec(g.at("v").data);
    const auto& exp = g.at("expected");
    const auto& exp_lse = g.at("lse");

    DeviceBuffer<engine::half> d_q(q);
    DeviceBuffer<engine::half> d_k(k);
    DeviceBuffer<engine::half> d_v(v);
    DeviceBuffer<engine::half> d_out(usize(c.s_q * c.n_heads * c.head_dim));
    DeviceBuffer<float> d_lse(usize(c.s_q * c.n_heads));
    d_out.zero();
    d_lse.zero();

    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    fn(d_q.get(), d_k.get(), d_v.get(), d_out.get(), c.s_q, c.s_kv, c.n_heads,
       c.n_kv_heads, c.head_dim, c.s_kv * c.head_dim, scale, c.q_pos_offset, c.is_causal,
       d_lse.get(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host_out = to_float_vec(d_out.download());
    const auto host_lse = d_lse.download();
    CHECK_CASE(std::string(label) + " " + stem, host_out.data(), exp.data.data(),
               host_out.size(), attention_rtol(), attention_atol(c.s_kv));
    CHECK_CASE(std::string(label) + " " + stem + " (lse)", host_lse.data(),
               exp_lse.data.data(), host_lse.size(), lse_rtol(), lse_atol(c.s_kv));
  }
}

void verify_prefill_attention_contracts(engtest::TestContext& ctx, PrefillAttentionFn fn,
                                        const char* label) {
  const std::int64_t D = 64, H_q = 4, H_kv = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const engine::half h_poison = engine::float_to_half(7.5f);

  // s_q == 0 is a legal no-op.
  {
    std::vector<engine::half> init(usize(2 * H_q * D), h_poison);
    DeviceBuffer<engine::half> d_buf(init);
    fn(d_buf.get(), d_buf.get(), d_buf.get(), d_buf.get(), 0, 4, H_q, H_kv, D, 4 * D,
       scale, 0, true, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (engine::half h : d_buf.download()) EXPECT_EQ(engine::half_to_float(h), 7.5f);
  }

  // s_kv == 0 zero-fills output and writes -inf LSE.
  {
    const std::int64_t s_q = 3;
    std::vector<engine::half> init(usize(s_q * H_q * D), h_poison);
    DeviceBuffer<engine::half> d_in(init);
    DeviceBuffer<engine::half> d_out(init);
    DeviceBuffer<float> d_lse(std::vector<float>(usize(s_q * H_q), 42.0f));
    fn(d_in.get(), d_in.get(), d_in.get(), d_out.get(), s_q, 0, H_q, H_kv, D, 0, scale, 0,
       true, d_lse.get(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (engine::half h : d_out.download()) EXPECT_EQ(engine::half_to_float(h), 0.0f);
    for (float val : d_lse.download()) EXPECT_TRUE(std::isinf(val) && val < 0.0f);
  }

  // Non-contiguous KV cache stride > s_kv * D skips poisoned inactive tokens.
  {
    LOAD_GOLDEN(g, "attention__s17_h8_kv2_d64_causal");
    const auto q = to_half_vec(g.at("q").data);
    const auto k_pack = to_half_vec(g.at("k").data);
    const auto v_pack = to_half_vec(g.at("v").data);
    const std::int64_t s_q = 17, s_kv = 17, hq = 8, hkv = 2, max_seq = 40;
    const engine::half h_nan =
        engine::float_to_half(std::numeric_limits<float>::quiet_NaN());

    std::vector<engine::half> k_cache(usize(hkv * max_seq * D), h_nan);
    std::vector<engine::half> v_cache(usize(hkv * max_seq * D), h_nan);
    for (std::int64_t hk = 0; hk < hkv; ++hk) {
      std::copy_n(k_pack.data() + hk * s_kv * D, usize(s_kv * D),
                  k_cache.data() + hk * max_seq * D);
      std::copy_n(v_pack.data() + hk * s_kv * D, usize(s_kv * D),
                  v_cache.data() + hk * max_seq * D);
    }

    DeviceBuffer<engine::half> d_q(q), d_k(k_cache), d_v(v_cache), d_out(q.size());
    d_out.zero();
    fn(d_q.get(), d_k.get(), d_v.get(), d_out.get(), s_q, s_kv, hq, hkv, D, max_seq * D,
       scale, 0, true, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host = to_float_vec(d_out.download());
    CHECK_CASE(std::string(label) + " strided KV cache", host.data(),
               g.at("expected").data.data(), host.size(), attention_rtol(),
               attention_atol(s_kv));
  }

  // Parameter validation errors.
  DeviceBuffer<engine::half> d(usize(H_q * 4 * D));
  d.zero();
  engine::half* p = d.get();
  EXPECT_THROWS(
      fn(nullptr, p, p, p, 1, 2, H_q, H_kv, D, 2 * D, scale, 0, true, nullptr, nullptr));
  EXPECT_THROWS(fn(p, p, p, p, 1, 2, 6, 4, D, 2 * D, scale, 0, true, nullptr, nullptr));
  EXPECT_THROWS(
      fn(p, p, p, p, 1, 2, H_q, H_kv, 32, 2 * 32, scale, 0, true, nullptr, nullptr));
  EXPECT_THROWS(
      fn(p, p, p, p, 1, 4, H_q, H_kv, D, 2 * D, scale, 0, true, nullptr, nullptr));
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

  using MatmulFn = void (*)(const float*, const float*, float*, std::int64_t,
                            std::int64_t, std::int64_t, cudaStream_t);
  const MatmulFn fns[] = {engine::cuda::matmul_naive, engine::cuda::matmul_tiled,
                          engine::cuda::matmul_register_tiled};
  for (MatmulFn fn : fns) {
    d_c.upload(poison);
    fn(d_a.get(), d_b.get(), d_c.get(), M, N, /*K=*/0, /*stream=*/nullptr);
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

  LOAD_GOLDEN(g, "rmsnorm_linear__32x256x512_now");
  const auto& in = g.at("input");
  const auto& W = g.at("W");
  const std::int64_t M = 32, N = 256, K = 512;

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

TEST(kernels, rmsnorm_linear_multi_stream_pool_safety) {
  REQUIRE_CUDA_DEVICE();

  // Exercise both decode (M = 1) and prefill (M = 32) scratch paths concurrently
  // across two non-blocking streams so PoolAllocator must defer scratch reuse.
  engine::CudaStream stream_a;
  engine::CudaStream stream_b;

  constexpr std::int64_t K = 512;
  constexpr std::int64_t N = 512;

  struct Run {
    std::int64_t m;
    DeviceBuffer<float> d_in;
    DeviceBuffer<float> d_weight;
    DeviceBuffer<float> d_w;
    DeviceBuffer<float> d_out;
    std::vector<float> expected;
  };

  std::vector<Run> runs;
  runs.reserve(6);
  for (int i = 0; i < 6; ++i) {
    const std::int64_t M = (i % 2 == 0) ? 32 : 1;
    const float seed = static_cast<float>(i + 1) * 1.37f;

    std::vector<float> in(usize(M * K));
    std::vector<float> weight(usize(K));
    std::vector<float> w(usize(K * N));
    for (std::size_t idx = 0; idx < in.size(); ++idx) {
      in[idx] = std::sin(seed + static_cast<float>(idx) * 0.013f);
    }
    for (std::size_t idx = 0; idx < weight.size(); ++idx) {
      weight[idx] = 0.75f + 0.25f * std::cos(seed + static_cast<float>(idx) * 0.031f);
    }
    for (std::size_t idx = 0; idx < w.size(); ++idx) {
      w[idx] = 0.05f * std::cos(seed * 1.7f + static_cast<float>(idx) * 0.007f);
    }

    std::vector<float> expected(usize(M * N), 0.0f);
    engine::cpu::rmsnorm_linear(in.data(), weight.data(), w.data(), expected.data(), M, N,
                                K, kEps);
    runs.emplace_back();
    Run& run = runs.back();
    run.m = M;
    run.d_in = DeviceBuffer<float>(in);
    run.d_weight = DeviceBuffer<float>(weight);
    run.d_w = DeviceBuffer<float>(w);
    run.d_out = DeviceBuffer<float>(usize(M * N));
    run.d_out.zero();
    run.expected = std::move(expected);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (std::size_t i = 0; i < runs.size(); ++i) {
    Run& r = runs[i];
    cudaStream_t stream = (i % 2 == 0) ? stream_a.get() : stream_b.get();
    engine::cuda::rmsnorm_linear(r.d_in.get(), r.d_weight.get(), r.d_w.get(),
                                 r.d_out.get(), r.m, N, K, kEps, stream);
  }
  stream_a.synchronize();
  stream_b.synchronize();

  for (std::size_t i = 0; i < runs.size(); ++i) {
    const std::vector<float> host = runs[i].d_out.download();
    CHECK_CASE("rmsnorm_linear multi-stream #" + std::to_string(i), host.data(),
               runs[i].expected.data(), host.size(), kMatmulRtol, kMatmulAtol);
  }
}

//===----------------------------------------------------------------------===//
// Exercise 9 (GEMV) -- Matrix-Vector product for decode token generation.
//===----------------------------------------------------------------------===//

struct GemvShape {
  std::int64_t n, k;
};

const GemvShape kGemvShapes[] = {
    {256, 4096},
    {768, 256},
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

TEST(kernels, gemv_and_gemv_fp16_with_k_zero_zero_output) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t N = 13;
  {
    const std::vector<float> poison(usize(N), 7.0f);
    DeviceBuffer<float> d_A(1);
    DeviceBuffer<float> d_x(1);
    DeviceBuffer<float> d_out(poison);
    d_A.zero();
    d_x.zero();

    engine::cuda::gemv(d_A.get(), d_x.get(), d_out.get(), N, /*K=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (float v : d_out.download()) {
      EXPECT_EQ(v, 0.0f);
    }
  }
  {
    const std::vector<engine::half> poison(usize(N), engine::float_to_half(7.0f));
    DeviceBuffer<engine::half> d_A(1);
    DeviceBuffer<engine::half> d_x(1);
    DeviceBuffer<engine::half> d_out(poison);
    d_A.zero();
    d_x.zero();

    engine::cuda::gemv_fp16(d_A.get(), d_x.get(), d_out.get(), N, /*K=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (engine::half h : d_out.download()) {
      EXPECT_EQ(engine::half_to_float(h), 0.0f);
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase 3 -- FP16 and missing inference kernels: gemv_fp16, embedding, argmax
//===----------------------------------------------------------------------===//

TEST(kernels, gemv_fp16_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  struct Shape {
    std::int64_t n, k;
  };
  const Shape shapes[] = {
      {64, 64},
      {128, 256},
      {73, 125},  // unaligned fallback
      {256, 4096},
  };

  for (const auto& s : shapes) {
    const std::string stem =
        "gemv_fp16__" + std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);

    const auto h_A = to_half_vec(g.at("A").data);
    const auto h_x = to_half_vec(g.at("x").data);
    const auto& exp = g.at("expected");

    DeviceBuffer<engine::half> d_A(h_A);
    DeviceBuffer<engine::half> d_x(h_x);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(s.n));
    d_out.zero();

    engine::cuda::gemv_fp16(d_A.get(), d_x.get(), d_out.get(), s.n, s.k);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host_f = to_float_vec(d_out.download());
    CHECK_CASE(stem, host_f.data(), exp.data.data(), host_f.size(), 2e-3, 2e-3);
  }
}

TEST(kernels, embedding_f32_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  struct Case {
    std::int64_t v, d, t;
  };
  const Case cases[] = {
      {50, 16, 4},
      {1000, 256, 5},  // vectorized aligned width
      {100, 65, 3},    // unaligned fallback
  };

  for (const auto& c : cases) {
    const std::string stem = "embedding__" + std::to_string(c.t) + "x" +
                             std::to_string(c.d) + "_v" + std::to_string(c.v);
    LOAD_GOLDEN(g, stem);

    const auto& table = g.at("table");
    const auto ids = to_i32_vec(g.at("ids").data);
    const auto& exp = g.at("expected");

    DeviceBuffer<float> d_table(table.data);
    DeviceBuffer<std::int32_t> d_ids(ids);
    DeviceBuffer<float> d_out(static_cast<std::size_t>(c.t * c.d));
    d_out.zero();

    engine::cuda::embedding(d_table.get(), d_ids.get(), d_out.get(), c.t, c.d, c.v);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), exp.data.data(), host.size(), kAddRtol, kAddAtol);
  }
}

TEST(kernels, embedding_fp16_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  struct Case {
    std::int64_t v, d, t;
  };
  const Case cases[] = {
      {50, 16, 4},
      {1000, 256, 5},
      {100, 65, 3},
  };

  for (const auto& c : cases) {
    const std::string stem = "embedding_fp16__" + std::to_string(c.t) + "x" +
                             std::to_string(c.d) + "_v" + std::to_string(c.v);
    LOAD_GOLDEN(g, stem);

    const auto table = to_half_vec(g.at("table").data);
    const auto ids = to_i32_vec(g.at("ids").data);
    const auto& exp = g.at("expected");

    DeviceBuffer<engine::half> d_table(table);
    DeviceBuffer<std::int32_t> d_ids(ids);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(c.t * c.d));
    d_out.zero();

    engine::cuda::embedding_fp16(d_table.get(), d_ids.get(), d_out.get(), c.t, c.d, c.v);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host_f = to_float_vec(d_out.download());
    CHECK_CASE(stem, host_f.data(), exp.data.data(), host_f.size(), kAddRtol, kAddAtol);
  }
}

TEST(kernels, argmax_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  auto check_stem = [&](const std::string& stem) {
    LOAD_GOLDEN(g, stem);
    const auto& logits = g.at("logits");
    const auto expected_idx =
        static_cast<std::int32_t>(std::lround(g.at("expected").data[0]));
    const std::int64_t vocab_size = static_cast<std::int64_t>(logits.data.size());

    DeviceBuffer<float> d_logits(logits.data);
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});

    engine::cuda::argmax(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], expected_idx);
  };

  for (std::int64_t v : {1, 31, 512, 1025, 32000}) {
    check_stem("argmax__v" + std::to_string(v));
  }
  check_stem("argmax__tie_negative");
}

TEST(kernels, argmax_fp16_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  auto check_stem = [&](const std::string& stem) {
    LOAD_GOLDEN(g, stem);
    const auto logits = to_half_vec(g.at("logits").data);
    const auto expected_idx =
        static_cast<std::int32_t>(std::lround(g.at("expected").data[0]));
    const std::int64_t vocab_size = static_cast<std::int64_t>(logits.size());

    DeviceBuffer<engine::half> d_logits(logits);
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});

    engine::cuda::argmax_fp16(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], expected_idx);
  };

  for (std::int64_t v : {1, 31, 512, 1025, 32000}) {
    check_stem("argmax_fp16__v" + std::to_string(v));
  }
  check_stem("argmax_fp16__tie_negative");
}

TEST(kernels, argmax_all_neg_inf_returns_zero) {
  REQUIRE_CUDA_DEVICE();

  const float neg_inf = -std::numeric_limits<float>::infinity();
  const std::int64_t vocab_size = 1025;  // spans > 1 stride across all 512 threads

  // All -inf logits must return index 0 (never the old 0x7fffffff sentinel).
  {
    std::vector<float> logits(static_cast<std::size_t>(vocab_size), neg_inf);
    DeviceBuffer<float> d_logits(logits);
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});
    engine::cuda::argmax(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], 0);
  }

  // -inf mixed with finite values < -1e38f must still select the lowest-index winner.
  {
    std::vector<float> logits(static_cast<std::size_t>(vocab_size), neg_inf);
    logits[700] = -2e38f;
    logits[900] = -2e38f;
    DeviceBuffer<float> d_logits(logits);
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});
    engine::cuda::argmax(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], 700);
  }

  // Single-element -inf edge case.
  {
    DeviceBuffer<float> d_logits(std::vector<float>{neg_inf});
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});
    engine::cuda::argmax(d_logits.get(), d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], 0);
  }

  // FP16 all -inf and mixed -inf / finite negative tie-breaking.
  {
    const engine::half h_neg_inf = engine::float_to_half(neg_inf);
    std::vector<engine::half> logits(static_cast<std::size_t>(vocab_size), h_neg_inf);
    DeviceBuffer<engine::half> d_logits(logits);
    DeviceBuffer<std::int32_t> d_out(std::vector<std::int32_t>{-1});
    engine::cuda::argmax_fp16(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], 0);

    d_out.upload(std::vector<std::int32_t>{-1});
    logits[611] = engine::float_to_half(-50000.0f);
    logits[950] = engine::float_to_half(-50000.0f);
    d_logits.upload(logits);
    engine::cuda::argmax_fp16(d_logits.get(), d_out.get(), vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(d_out.download()[0], 611);
  }
}

//===----------------------------------------------------------------------===//
// Phase 4 -- Fused SwiGLU activation tests
//===----------------------------------------------------------------------===//

TEST(kernels, swiglu_f32_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t sizes[] = {1, 65, 1024, 4096, 11008};
  for (std::int64_t n : sizes) {
    const std::string stem = "swiglu__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);

    const auto& gate = g.at("gate");
    const auto& up = g.at("up");
    const auto& exp = g.at("expected");

    DeviceBuffer<float> d_gate(gate.data);
    DeviceBuffer<float> d_up(up.data);
    DeviceBuffer<float> d_out(static_cast<std::size_t>(n));
    d_out.zero();

    engine::cuda::swiglu(d_gate.get(), d_up.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host = d_out.download();
    CHECK_CASE(stem, host.data(), exp.data.data(), host.size(), 1e-5, 1e-5);
  }
}

TEST(kernels, swiglu_fp16_matches_reference) {
  REQUIRE_CUDA_DEVICE();

  const std::int64_t sizes[] = {1, 65, 1024, 4096, 11008};
  for (std::int64_t n : sizes) {
    const std::string stem = "swiglu_fp16__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);

    const auto gate = to_half_vec(g.at("gate").data);
    const auto up = to_half_vec(g.at("up").data);
    const auto& exp = g.at("expected");

    DeviceBuffer<engine::half> d_gate(gate);
    DeviceBuffer<engine::half> d_up(up);
    DeviceBuffer<engine::half> d_out(static_cast<std::size_t>(n));
    d_out.zero();

    engine::cuda::swiglu_fp16(d_gate.get(), d_up.get(), d_out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host_f = to_float_vec(d_out.download());
    CHECK_CASE(stem, host_f.data(), exp.data.data(), host_f.size(), 2e-3, 2e-3);
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
  EXPECT_NO_THROW(engine::cuda::gemv_fp16(p_h, p_h, p_h, 0, 8));
  EXPECT_NO_THROW(engine::cuda::embedding(p, p_i, p, 0, 8, 10));
  EXPECT_NO_THROW(engine::cuda::embedding(p, p_i, p, 8, 0, 10));
  EXPECT_NO_THROW(engine::cuda::embedding_fp16(p_h, p_i, p_h, 0, 8, 10));
  EXPECT_NO_THROW(engine::cuda::embedding_fp16(p_h, p_i, p_h, 8, 0, 10));
  EXPECT_NO_THROW(engine::cuda::matmul_register_tiled(p, p, p, 0, 8, 8));
  EXPECT_NO_THROW(engine::cuda::matmul_register_tiled(p, p, p, 8, 0, 8));
  EXPECT_NO_THROW(engine::cuda::swiglu(p, p, p, 0));
  EXPECT_NO_THROW(engine::cuda::swiglu_fp16(p_h, p_h, p_h, 0));

  // And nothing may have been launched, let alone written.
  CUDA_CHECK(cudaDeviceSynchronize());
  for (float v : d.download()) EXPECT_EQ(v, 0.0f);
}

//===----------------------------------------------------------------------===//
// Module 4 -- Attention, RoPE, FlashAttention-2, and FlashDecoding
//===----------------------------------------------------------------------===//

TEST(kernels, attention_naive_matches_reference) {
  REQUIRE_CUDA_DEVICE();
  verify_prefill_attention_against_golden(ctx, engine::cuda::attention_naive, "naive");
}

TEST(kernels, attention_naive_contracts_and_edge_cases) {
  REQUIRE_CUDA_DEVICE();
  verify_prefill_attention_contracts(ctx, engine::cuda::attention_naive, "naive");
}

TEST(kernels, rope_matches_reference) {
  REQUIRE_CUDA_DEVICE();
  for (const auto& c : kRopeCases) {
    const std::string stem = std::string("rope__") + c.name;
    LOAD_GOLDEN(g, stem);

    const auto q_in = to_half_vec(g.at("q_in").data);
    const auto k_in = to_half_vec(g.at("k_in").data);
    const auto& exp_q = g.at("expected").data;
    const auto& exp_k = g.at("k_out").data;

    const std::int64_t half_d = c.head_dim / 2;
    const std::int64_t max_seq = c.pos_offset + c.s;
    std::vector<float> cos_table(usize(max_seq * half_d));
    std::vector<float> sin_table(usize(max_seq * half_d));
    engine::cpu::rope_precompute_tables(cos_table.data(), sin_table.data(), max_seq,
                                        c.head_dim, 10000.0f);

    DeviceBuffer<engine::half> d_q(q_in);
    DeviceBuffer<engine::half> d_k(k_in);
    DeviceBuffer<float> d_cos(cos_table);
    DeviceBuffer<float> d_sin(sin_table);

    engine::cuda::rope(d_q.get(), d_k.get(), d_cos.get(), d_sin.get(), c.s, c.n_heads,
                       c.n_kv_heads, c.head_dim, c.pos_offset, c.convention);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto q_f = to_float_vec(d_q.download());
    const auto k_f = to_float_vec(d_k.download());
    CHECK_CASE(stem + " (q)", q_f.data(), exp_q.data(), q_f.size(), attention_rtol(),
               attention_atol(1));
    CHECK_CASE(stem + " (k)", k_f.data(), exp_k.data(), k_f.size(), attention_rtol(),
               attention_atol(1));
  }
}

TEST(kernels, rope_contracts_and_edge_cases) {
  REQUIRE_CUDA_DEVICE();
  const engine::half h_val = engine::float_to_half(2.5f);
  DeviceBuffer<engine::half> d_q(std::vector<engine::half>(64, h_val));
  DeviceBuffer<float> d_tab(std::vector<float>(64, 1.0f));
  engine::cuda::rope(d_q.get(), nullptr, d_tab.get(), d_tab.get(), 0, 1, 0, 64);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(engine::half_to_float(d_q.download()[0]), 2.5f);

  engine::cuda::rope(d_q.get(), nullptr, d_tab.get(), d_tab.get(), 1, 1, 0, 64);
  CUDA_CHECK(cudaDeviceSynchronize());

  EXPECT_THROWS(
      engine::cuda::rope(d_q.get(), nullptr, d_tab.get(), d_tab.get(), 1, 1, 1, 64));
  EXPECT_THROWS(
      engine::cuda::rope(d_q.get(), d_q.get(), d_tab.get(), d_tab.get(), 1, 1, 1, 48));
  EXPECT_THROWS(engine::cuda::rope(d_q.get(), d_q.get(), d_tab.get(), d_tab.get(), 1, 1,
                                   1, 64, -1));
}

TEST(kernels, attention_prefill_matches_reference) {
  REQUIRE_CUDA_DEVICE();
  verify_prefill_attention_against_golden(ctx, engine::cuda::attention_prefill, "flash");
  auto per_head_fn = [](const engine::half* q, const engine::half* k,
                        const engine::half* v, engine::half* out, std::int64_t s_q,
                        std::int64_t s_kv, std::int64_t n_heads, std::int64_t n_kv_heads,
                        std::int64_t head_dim, std::int64_t kv_head_stride, float scale,
                        std::int64_t q_pos_offset, bool is_causal, float* out_lse,
                        cudaStream_t stream) {
    engine::cuda::detail::attention_prefill_with_mode(
        q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim, kv_head_stride, scale,
        q_pos_offset, is_causal, out_lse, /*share_gqa_kv=*/false, stream);
  };
  verify_prefill_attention_against_golden(ctx, per_head_fn, "flash_no_gqa_share");
}

TEST(kernels, attention_prefill_agrees_with_naive_and_online) {
  REQUIRE_CUDA_DEVICE();
  for (const auto& c : kAttentionCases) {
    const std::string stem = std::string("attention__") + c.name;
    LOAD_GOLDEN(g, stem);
    const auto q = to_half_vec(g.at("q").data);
    const auto k = to_half_vec(g.at("k").data);
    const auto v = to_half_vec(g.at("v").data);

    DeviceBuffer<engine::half> d_q(q), d_k(k), d_v(v);
    DeviceBuffer<engine::half> d_naive(q.size()), d_flash(q.size());
    d_naive.zero();
    d_flash.zero();

    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    const std::int64_t kv_stride = c.s_kv * c.head_dim;
    engine::cuda::attention_naive(d_q.get(), d_k.get(), d_v.get(), d_naive.get(), c.s_q,
                                  c.s_kv, c.n_heads, c.n_kv_heads, c.head_dim, kv_stride,
                                  scale, c.q_pos_offset, c.is_causal);
    engine::cuda::attention_prefill(d_q.get(), d_k.get(), d_v.get(), d_flash.get(), c.s_q,
                                    c.s_kv, c.n_heads, c.n_kv_heads, c.head_dim,
                                    kv_stride, scale, c.q_pos_offset, c.is_causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<engine::half> h_online(q.size());
    engine::cpu::attention_online(q.data(), k.data(), v.data(), h_online.data(), c.s_q,
                                  c.s_kv, c.n_heads, c.n_kv_heads, c.head_dim, kv_stride,
                                  scale, c.q_pos_offset, c.is_causal);

    const auto naive_f = to_float_vec(d_naive.download());
    const auto flash_f = to_float_vec(d_flash.download());
    const auto online_f = to_float_vec(h_online);
    CHECK_CASE(stem + " (flash vs naive)", flash_f.data(), naive_f.data(), flash_f.size(),
               attention_rtol(), attention_atol(c.s_kv));
    CHECK_CASE(stem + " (flash vs online_cpu)", flash_f.data(), online_f.data(),
               flash_f.size(), attention_rtol(), attention_atol(c.s_kv));
  }
}

TEST(kernels, attention_prefill_contracts_and_edge_cases) {
  REQUIRE_CUDA_DEVICE();
  verify_prefill_attention_contracts(ctx, engine::cuda::attention_prefill, "flash");
}

TEST(kernels, attention_decode_matches_reference) {
  REQUIRE_CUDA_DEVICE();
  for (const auto& c : kAttentionCases) {
    if (c.s_q != 1) continue;
    const std::string stem = std::string("attention__") + c.name;
    LOAD_GOLDEN(g, stem);
    const auto q = to_half_vec(g.at("q").data);
    const auto k = to_half_vec(g.at("k").data);
    const auto v = to_half_vec(g.at("v").data);
    const auto& exp = g.at("expected");

    DeviceBuffer<engine::half> d_q(q), d_k(k), d_v(v), d_out(q.size());
    d_out.zero();
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out.get(), c.s_kv,
                                   c.n_heads, c.n_kv_heads, c.head_dim,
                                   c.s_kv * c.head_dim, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto out_f = to_float_vec(d_out.download());
    CHECK_CASE("decode " + stem, out_f.data(), exp.data.data(), out_f.size(),
               attention_rtol(), attention_atol(c.s_kv));
  }
}

TEST(kernels, attention_decode_splits_and_contracts) {
  REQUIRE_CUDA_DEVICE();
  // Odd kv_len=513, D=128, GQA 4:1 across splits=1 vs splits=8 and edge kv_len < splits.
  LOAD_GOLDEN(g, "attention__s1_kv513_h8_kv2_d128_decode");
  const auto q = to_half_vec(g.at("q").data);
  const auto k = to_half_vec(g.at("k").data);
  const auto v = to_half_vec(g.at("v").data);
  const auto& exp = g.at("expected");
  const std::int64_t kv_len = 513, hq = 8, hkv = 2, D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  DeviceBuffer<engine::half> d_q(q), d_k(k), d_v(v), d_s1(q.size()), d_s8(q.size());
  d_s1.zero();
  d_s8.zero();
  engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_s1.get(), kv_len, hq,
                                 hkv, D, kv_len * D, scale, /*split_kv=*/1);
  engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_s8.get(), kv_len, hq,
                                 hkv, D, kv_len * D, scale, /*split_kv=*/8);
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto f1 = to_float_vec(d_s1.download());
  const auto f8 = to_float_vec(d_s8.download());
  CHECK_CASE("decode split_kv=1 vs golden", f1.data(), exp.data.data(), f1.size(),
             attention_rtol(), attention_atol(kv_len));
  CHECK_CASE("decode split_kv=8 vs golden", f8.data(), exp.data.data(), f8.size(),
             attention_rtol(), attention_atol(kv_len));
  CHECK_CASE("decode split_kv=1 vs 8", f1.data(), f8.data(), f1.size(), attention_rtol(),
             attention_atol(kv_len));

  // kv_len=3 < split_kv=16 edge case & kv_len == 0 zero-fill.
  DeviceBuffer<engine::half> d_out3(q.size()), d_out0(q.size());
  std::vector<engine::half> h_ref3(q.size());
  engine::cpu::attention(q.data(), k.data(), v.data(), h_ref3.data(), 1, 3, hq, hkv, D,
                         kv_len * D, scale, 0, false);
  d_out3.zero();
  engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out3.get(), 3, hq,
                                 hkv, D, kv_len * D, scale, /*split_kv=*/16);
  d_out0.upload(std::vector<engine::half>(q.size(), engine::float_to_half(5.0f)));
  engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out0.get(), 0, hq,
                                 hkv, D, 0, scale, /*split_kv=*/4);
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto f3 = to_float_vec(d_out3.download());
  const auto exp3 = to_float_vec(h_ref3);
  CHECK_CASE("decode kv_len=3 < split_kv=16", f3.data(), exp3.data(), f3.size(),
             attention_rtol(), attention_atol(3));
  for (engine::half h : d_out0.download()) EXPECT_EQ(engine::half_to_float(h), 0.0f);

  // kv_len == 1 exact single-key decode & per-head GQA ablation mode check.
  DeviceBuffer<engine::half> d_out1(q.size()), d_no_share(q.size());
  std::vector<engine::half> h_ref1(q.size());
  engine::cpu::attention(q.data(), k.data(), v.data(), h_ref1.data(), 1, 1, hq, hkv, D,
                         kv_len * D, scale, 0, false);
  d_out1.zero();
  d_no_share.zero();
  engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out1.get(), 1, hq,
                                 hkv, D, kv_len * D, scale, /*split_kv=*/0);
  engine::cuda::detail::attention_decode_with_mode(
      d_q.get(), d_k.get(), d_v.get(), d_no_share.get(), kv_len, hq, hkv, D, kv_len * D,
      scale, /*split_kv=*/8, /*share_gqa_kv=*/false, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CASE("decode kv_len=1", to_float_vec(d_out1.download()).data(),
             to_float_vec(h_ref1).data(), q.size(), attention_rtol(), attention_atol(1));
  CHECK_CASE("decode per-head split_kv=8 vs golden",
             to_float_vec(d_no_share.download()).data(), exp.data.data(), q.size(),
             attention_rtol(), attention_atol(kv_len));

  // Multi-stream ScopedPoolBuffer reuse across concurrent Split-KV decode launches.
  cudaStream_t stream_a = nullptr, stream_b = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking));
  DeviceBuffer<engine::half> d_out_a(q.size()), d_out_b(q.size());
  d_out_a.zero();
  d_out_b.zero();
  std::vector<engine::half> h_ref257(q.size());
  engine::cpu::attention(q.data(), k.data(), v.data(), h_ref257.data(), 1, 257, hq, hkv,
                         D, kv_len * D, scale, 0, false);
  for (int rep = 0; rep < 4; ++rep) {
    engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out_a.get(), kv_len,
                                   hq, hkv, D, kv_len * D, scale, /*split_kv=*/8,
                                   stream_a);
    engine::cuda::attention_decode(d_q.get(), d_k.get(), d_v.get(), d_out_b.get(), 257,
                                   hq, hkv, D, kv_len * D, scale, /*split_kv=*/8,
                                   stream_b);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  CUDA_CHECK(cudaStreamSynchronize(stream_b));
  CUDA_CHECK(cudaStreamDestroy(stream_a));
  CUDA_CHECK(cudaStreamDestroy(stream_b));
  CHECK_CASE("decode multi-stream A", to_float_vec(d_out_a.download()).data(),
             exp.data.data(), q.size(), attention_rtol(), attention_atol(kv_len));
  CHECK_CASE("decode multi-stream B", to_float_vec(d_out_b.download()).data(),
             to_float_vec(h_ref257).data(), q.size(), attention_rtol(),
             attention_atol(257));

  // Parameter contract rejections.
  engine::half* p = d_q.get();
  EXPECT_THROWS(
      engine::cuda::attention_decode(nullptr, p, p, p, 4, hq, hkv, D, 4 * D, scale));
  EXPECT_THROWS(engine::cuda::attention_decode(p, p, p, p, 4, 6, 4, D, 4 * D, scale));
  EXPECT_THROWS(
      engine::cuda::attention_decode(p, p, p, p, 4, hq, hkv, 32, 4 * 32, scale));
  EXPECT_THROWS(
      engine::cuda::attention_decode(p, p, p, p, 4, hq, hkv, D, 4 * D, scale, -1));
}

namespace {

// Derived tolerance for the full Attention Sub-Block pipeline
// (RMSNorm[F32] -> QKV[F32->F16] -> RoPE[F16] -> Attention[F16] -> O[F16->F32] + Residual[F32]):
// Two binary16 truncation boundaries (Q/K/V and Attention output), with RMS ~ 1 and
// projection W_o spectral norm ~ 2 (Xavier 1/sqrt(d_model)), give componentwise error
// bounded well within 16 * eps16 (~1.56e-2) relative and 8 * eps16 (~7.8e-3) absolute.
double subblock_rtol() { return 16.0 * std::ldexp(1.0, -10); }
double subblock_atol() { return 8.0 * std::ldexp(1.0, -10); }

void run_attention_subblock_gpu(const float* d_x, const float* d_rms_w, const float* d_wq,
                                const float* d_wk, const float* d_wv, const float* d_wo,
                                const float* d_cos, const float* d_sin,
                                engine::half* d_k_cache, engine::half* d_v_cache,
                                float* d_out, std::int64_t s_q, std::int64_t past_len,
                                std::int64_t kv_max_seq, std::int64_t n_heads,
                                std::int64_t n_kv_heads, std::int64_t head_dim,
                                float eps) {
  const std::int64_t d_model = n_heads * head_dim;
  const std::int64_t d_kv = n_kv_heads * head_dim;
  const std::size_t q_elems = static_cast<std::size_t>(s_q * d_model);
  const std::size_t kv_elems = static_cast<std::size_t>(s_q * d_kv);

  DeviceBuffer<float> d_x_norm(q_elems);
  DeviceBuffer<float> d_q_f32(q_elems);
  DeviceBuffer<float> d_k_f32(kv_elems);
  DeviceBuffer<float> d_v_f32(kv_elems);
  DeviceBuffer<float> d_attn_f32(q_elems);
  DeviceBuffer<float> d_o_proj(q_elems);

  DeviceBuffer<engine::half> d_q_f16(q_elems);
  DeviceBuffer<engine::half> d_k_f16(kv_elems);
  DeviceBuffer<engine::half> d_v_f16(kv_elems);
  DeviceBuffer<engine::half> d_attn_f16(q_elems);

  engine::cuda::rmsnorm(d_x, d_rms_w, d_x_norm.get(), s_q, d_model, eps);

  if (s_q > 1) {
    engine::cuda::matmul_register_tiled(d_x_norm.get(), d_wq, d_q_f32.get(), s_q, d_model,
                                        d_model);
    engine::cuda::matmul_register_tiled(d_x_norm.get(), d_wk, d_k_f32.get(), s_q, d_kv,
                                        d_model);
    engine::cuda::matmul_register_tiled(d_x_norm.get(), d_wv, d_v_f32.get(), s_q, d_kv,
                                        d_model);
  } else {
    engine::cuda::gemv(d_wq, d_x_norm.get(), d_q_f32.get(), d_model, d_model);
    engine::cuda::gemv(d_wk, d_x_norm.get(), d_k_f32.get(), d_kv, d_model);
    engine::cuda::gemv(d_wv, d_x_norm.get(), d_v_f32.get(), d_kv, d_model);
  }

  engine::cuda::detail::cast_f32_to_f16(d_q_f32.get(), d_q_f16.get(), s_q * d_model);
  engine::cuda::detail::cast_f32_to_f16(d_k_f32.get(), d_k_f16.get(), s_q * d_kv);
  engine::cuda::detail::cast_f32_to_f16(d_v_f32.get(), d_v_f16.get(), s_q * d_kv);

  engine::cuda::rope(d_q_f16.get(), d_k_f16.get(), d_cos, d_sin, s_q, n_heads, n_kv_heads,
                     head_dim, past_len, engine::RopeConvention::RotateHalf);

  engine::cuda::detail::kv_cache_write(d_k_f16.get(), d_v_f16.get(), d_k_cache, d_v_cache,
                                       s_q, n_kv_heads, head_dim, past_len, kv_max_seq);

  const std::int64_t kv_len = past_len + s_q;
  const std::int64_t kv_head_stride = kv_max_seq * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  if (s_q > 1) {
    engine::cuda::attention_prefill(d_q_f16.get(), d_k_cache, d_v_cache, d_attn_f16.get(),
                                    s_q, kv_len, n_heads, n_kv_heads, head_dim,
                                    kv_head_stride, scale, past_len, /*is_causal=*/true);
  } else {
    engine::cuda::attention_decode(d_q_f16.get(), d_k_cache, d_v_cache, d_attn_f16.get(),
                                   kv_len, n_heads, n_kv_heads, head_dim, kv_head_stride,
                                   scale, /*split_kv=*/0);
  }

  engine::cuda::detail::cast_f16_to_f32(d_attn_f16.get(), d_attn_f32.get(),
                                        s_q * d_model);
  if (s_q > 1) {
    engine::cuda::matmul_register_tiled(d_attn_f32.get(), d_wo, d_o_proj.get(), s_q,
                                        d_model, d_model);
  } else {
    engine::cuda::gemv(d_wo, d_attn_f32.get(), d_o_proj.get(), d_model, d_model);
  }
  engine::cuda::vector_add(d_x, d_o_proj.get(), d_out, s_q * d_model);
}

void run_attention_subblock_cpu(const float* x, const float* rms_w, const float* wq,
                                const float* wk, const float* wv, const float* wo,
                                const float* cos_tbl, const float* sin_tbl,
                                engine::half* k_cache, engine::half* v_cache, float* out,
                                std::int64_t s_q, std::int64_t past_len,
                                std::int64_t kv_max_seq, std::int64_t n_heads,
                                std::int64_t n_kv_heads, std::int64_t head_dim,
                                float eps) {
  const std::int64_t d_model = n_heads * head_dim;
  const std::int64_t d_kv = n_kv_heads * head_dim;
  const std::size_t q_elems = static_cast<std::size_t>(s_q * d_model);
  const std::size_t kv_elems = static_cast<std::size_t>(s_q * d_kv);

  std::vector<float> x_norm(q_elems);
  std::vector<float> q_f32(q_elems), k_f32(kv_elems), v_f32(kv_elems);
  engine::cpu::rmsnorm(x, rms_w, x_norm.data(), s_q, d_model, eps);
  engine::cpu::matmul(x_norm.data(), wq, q_f32.data(), s_q, d_model, d_model);
  engine::cpu::matmul(x_norm.data(), wk, k_f32.data(), s_q, d_kv, d_model);
  engine::cpu::matmul(x_norm.data(), wv, v_f32.data(), s_q, d_kv, d_model);

  auto q_f16 = to_half_vec(q_f32);
  auto k_f16 = to_half_vec(k_f32);
  auto v_f16 = to_half_vec(v_f32);
  engine::cpu::rope(q_f16.data(), k_f16.data(), cos_tbl, sin_tbl, s_q, n_heads,
                    n_kv_heads, head_dim, past_len, engine::RopeConvention::RotateHalf);

  for (std::int64_t t = 0; t < s_q; ++t) {
    for (std::int64_t h = 0; h < n_kv_heads; ++h) {
      const std::size_t src_off =
          static_cast<std::size_t>((t * n_kv_heads + h) * head_dim);
      const std::size_t dst_off =
          static_cast<std::size_t>((h * kv_max_seq + past_len + t) * head_dim);
      for (std::int64_t d = 0; d < head_dim; ++d) {
        k_cache[dst_off + static_cast<std::size_t>(d)] =
            k_f16[src_off + static_cast<std::size_t>(d)];
        v_cache[dst_off + static_cast<std::size_t>(d)] =
            v_f16[src_off + static_cast<std::size_t>(d)];
      }
    }
  }

  std::vector<engine::half> attn_f16(q_elems);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  engine::cpu::attention(q_f16.data(), k_cache, v_cache, attn_f16.data(), s_q,
                         past_len + s_q, n_heads, n_kv_heads, head_dim,
                         kv_max_seq * head_dim, scale, past_len,
                         /*is_causal=*/(s_q > 1));

  const auto attn_f32 = to_float_vec(attn_f16);
  std::vector<float> o_proj(q_elems);
  engine::cpu::matmul(attn_f32.data(), wo, o_proj.data(), s_q, d_model, d_model);
  engine::cpu::vector_add(x, o_proj.data(), out, s_q * d_model);
}

std::vector<float> pseudo_normal_vec(std::size_t n, std::uint64_t seed, float scale) {
  std::vector<float> out(n);
  std::uint64_t s = seed;
  auto next_u32 = [&s]() -> std::uint32_t {
    s += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32);
  };
  for (std::size_t i = 0; i < n; ++i) {
    // Sum of 6 uniforms centered at 0 (~std 1/sqrt(2)) scaled cleanly without <random>.
    double acc = -3.0;
    for (int k = 0; k < 6; ++k) {
      acc += static_cast<double>(next_u32()) / 4294967296.0;
    }
    out[i] = static_cast<float>(acc * std::sqrt(2.0) * static_cast<double>(scale));
  }
  return out;
}

}  // namespace

TEST(kernels, attention_subblock_prefill_and_decode_golden) {
  REQUIRE_CUDA_DEVICE();
  struct SubBlockGolden {
    const char* stem;
    std::int64_t s_q, past_len, n_heads, n_kv_heads, head_dim;
  };
  const SubBlockGolden cases[] = {
      {"attn_subblock__prefill_s16_h8_kv2_d64", 16, 0, 8, 2, 64},
      {"attn_subblock__decode_s1_kv257_h8_kv2_d64", 1, 256, 8, 2, 64},
  };

  for (const auto& c : cases) {
    LOAD_GOLDEN(g, c.stem);
    const std::int64_t kv_max_seq = c.past_len + c.s_q;
    const std::int64_t d_model = c.n_heads * c.head_dim;

    DeviceBuffer<float> d_x(g.at("x").data);
    DeviceBuffer<float> d_rms_w(g.at("rms_w").data);
    DeviceBuffer<float> d_wq(g.at("w_q").data);
    DeviceBuffer<float> d_wk(g.at("w_k").data);
    DeviceBuffer<float> d_wv(g.at("w_v").data);
    DeviceBuffer<float> d_wo(g.at("w_o").data);
    DeviceBuffer<float> d_cos(g.at("cos").data);
    DeviceBuffer<float> d_sin(g.at("sin").data);
    DeviceBuffer<float> d_out(static_cast<std::size_t>(c.s_q * d_model));

    const std::size_t cache_elems =
        static_cast<std::size_t>(c.n_kv_heads * kv_max_seq * c.head_dim);
    std::vector<engine::half> h_k_cache(cache_elems, engine::float_to_half(0.0f));
    std::vector<engine::half> h_v_cache(cache_elems, engine::float_to_half(0.0f));
    if (c.past_len > 0) {
      const auto k_past = to_half_vec(g.at("k_past").data);
      const auto v_past = to_half_vec(g.at("v_past").data);
      for (std::int64_t h = 0; h < c.n_kv_heads; ++h) {
        const std::size_t src_off = static_cast<std::size_t>(h * c.past_len * c.head_dim);
        const std::size_t dst_off = static_cast<std::size_t>(h * kv_max_seq * c.head_dim);
        const std::size_t copy_len = static_cast<std::size_t>(c.past_len * c.head_dim);
        std::copy_n(k_past.begin() + static_cast<std::ptrdiff_t>(src_off), copy_len,
                    h_k_cache.begin() + static_cast<std::ptrdiff_t>(dst_off));
        std::copy_n(v_past.begin() + static_cast<std::ptrdiff_t>(src_off), copy_len,
                    h_v_cache.begin() + static_cast<std::ptrdiff_t>(dst_off));
      }
    }
    DeviceBuffer<engine::half> d_k_cache(h_k_cache);
    DeviceBuffer<engine::half> d_v_cache(h_v_cache);

    run_attention_subblock_gpu(
        d_x.get(), d_rms_w.get(), d_wq.get(), d_wk.get(), d_wv.get(), d_wo.get(),
        d_cos.get(), d_sin.get(), d_k_cache.get(), d_v_cache.get(), d_out.get(), c.s_q,
        c.past_len, kv_max_seq, c.n_heads, c.n_kv_heads, c.head_dim, /*eps=*/1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto got = d_out.download();
    const auto& exp = g.at("expected").data;
    CHECK_CASE(c.stem, got.data(), exp.data(), got.size(), subblock_rtol(),
               subblock_atol());
  }
}

TEST(kernels, attention_subblock_tinyllama_shape) {
  REQUIRE_CUDA_DEVICE();
  // Exact TinyLlama-1.1B attention sub-block dimensions: prefill (s_q=64) followed
  // sequentially by 2 decode steps (kv_len=65, 66) sharing the same KV cache.
  constexpr std::int64_t n_heads = 32;
  constexpr std::int64_t n_kv_heads = 4;
  constexpr std::int64_t head_dim = 64;
  constexpr std::int64_t d_model = n_heads * head_dim;  // 2048
  constexpr std::int64_t d_kv = n_kv_heads * head_dim;  // 256
  constexpr std::int64_t s_prefill = 64;
  constexpr std::int64_t kv_max_seq = 128;
  constexpr float eps = 1e-5f;

  const float inv_s = 1.0f / std::sqrt(static_cast<float>(d_model));
  std::vector<float> h_rms_w(static_cast<std::size_t>(d_model), 1.0f);
  const auto rms_delta = pseudo_normal_vec(static_cast<std::size_t>(d_model), 101u, 0.1f);
  for (std::size_t i = 0; i < h_rms_w.size(); ++i) h_rms_w[i] += rms_delta[i];

  const auto h_wq =
      pseudo_normal_vec(static_cast<std::size_t>(d_model * d_model), 102u, inv_s);
  const auto h_wk =
      pseudo_normal_vec(static_cast<std::size_t>(d_model * d_kv), 103u, inv_s);
  const auto h_wv =
      pseudo_normal_vec(static_cast<std::size_t>(d_model * d_kv), 104u, inv_s);
  const auto h_wo =
      pseudo_normal_vec(static_cast<std::size_t>(d_model * d_model), 105u, inv_s);

  std::vector<float> h_cos(static_cast<std::size_t>(kv_max_seq * (head_dim / 2)));
  std::vector<float> h_sin(static_cast<std::size_t>(kv_max_seq * (head_dim / 2)));
  engine::cpu::rope_precompute_tables(h_cos.data(), h_sin.data(), kv_max_seq, head_dim);

  const auto h_x0 =
      pseudo_normal_vec(static_cast<std::size_t>(s_prefill * d_model), 201u, 1.0f);
  const auto h_x1 = pseudo_normal_vec(static_cast<std::size_t>(d_model), 202u, 1.0f);
  const auto h_x2 = pseudo_normal_vec(static_cast<std::size_t>(d_model), 203u, 1.0f);

  DeviceBuffer<float> d_rms_w(h_rms_w), d_wq(h_wq), d_wk(h_wk), d_wv(h_wv), d_wo(h_wo);
  DeviceBuffer<float> d_cos(h_cos), d_sin(h_sin);
  const std::size_t cache_elems =
      static_cast<std::size_t>(n_kv_heads * kv_max_seq * head_dim);
  DeviceBuffer<engine::half> d_k_cache(cache_elems), d_v_cache(cache_elems);
  d_k_cache.zero();
  d_v_cache.zero();
  std::vector<engine::half> cpu_k_cache(cache_elems, engine::float_to_half(0.0f));
  std::vector<engine::half> cpu_v_cache(cache_elems, engine::float_to_half(0.0f));

  // Step 0: Prefill s_q=64.
  DeviceBuffer<float> d_x0(h_x0), d_out0(h_x0.size());
  std::vector<float> cpu_out0(h_x0.size());
  run_attention_subblock_gpu(
      d_x0.get(), d_rms_w.get(), d_wq.get(), d_wk.get(), d_wv.get(), d_wo.get(),
      d_cos.get(), d_sin.get(), d_k_cache.get(), d_v_cache.get(), d_out0.get(), s_prefill,
      /*past_len=*/0, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  run_attention_subblock_cpu(
      h_x0.data(), h_rms_w.data(), h_wq.data(), h_wk.data(), h_wv.data(), h_wo.data(),
      h_cos.data(), h_sin.data(), cpu_k_cache.data(), cpu_v_cache.data(), cpu_out0.data(),
      s_prefill, /*past_len=*/0, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto gpu_out0 = d_out0.download();
  CHECK_CASE("tinyllama subblock prefill s=64", gpu_out0.data(), cpu_out0.data(),
             gpu_out0.size(), subblock_rtol(), subblock_atol());

  // Step 1: Decode s_q=1 at past_len=64 (kv_len=65), reusing prefill KV cache.
  DeviceBuffer<float> d_x1(h_x1), d_out1(h_x1.size());
  std::vector<float> cpu_out1(h_x1.size());
  run_attention_subblock_gpu(
      d_x1.get(), d_rms_w.get(), d_wq.get(), d_wk.get(), d_wv.get(), d_wo.get(),
      d_cos.get(), d_sin.get(), d_k_cache.get(), d_v_cache.get(), d_out1.get(), /*s_q=*/1,
      /*past_len=*/64, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  run_attention_subblock_cpu(
      h_x1.data(), h_rms_w.data(), h_wq.data(), h_wk.data(), h_wv.data(), h_wo.data(),
      h_cos.data(), h_sin.data(), cpu_k_cache.data(), cpu_v_cache.data(), cpu_out1.data(),
      /*s_q=*/1, /*past_len=*/64, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto gpu_out1 = d_out1.download();
  CHECK_CASE("tinyllama subblock decode step1 kv=65", gpu_out1.data(), cpu_out1.data(),
             gpu_out1.size(), subblock_rtol(), subblock_atol());

  // Step 2: Decode s_q=1 at past_len=65 (kv_len=66), verifying KV cache growth.
  DeviceBuffer<float> d_x2(h_x2), d_out2(h_x2.size());
  std::vector<float> cpu_out2(h_x2.size());
  run_attention_subblock_gpu(
      d_x2.get(), d_rms_w.get(), d_wq.get(), d_wk.get(), d_wv.get(), d_wo.get(),
      d_cos.get(), d_sin.get(), d_k_cache.get(), d_v_cache.get(), d_out2.get(), /*s_q=*/1,
      /*past_len=*/65, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  run_attention_subblock_cpu(
      h_x2.data(), h_rms_w.data(), h_wq.data(), h_wk.data(), h_wv.data(), h_wo.data(),
      h_cos.data(), h_sin.data(), cpu_k_cache.data(), cpu_v_cache.data(), cpu_out2.data(),
      /*s_q=*/1, /*past_len=*/65, kv_max_seq, n_heads, n_kv_heads, head_dim, eps);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto gpu_out2 = d_out2.download();
  CHECK_CASE("tinyllama subblock decode step2 kv=66", gpu_out2.data(), cpu_out2.data(),
             gpu_out2.size(), subblock_rtol(), subblock_atol());
}

#endif  // ENGINE_HAS_CUDA
