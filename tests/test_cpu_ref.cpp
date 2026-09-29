//===----------------------------------------------------------------------===//
// tests/test_cpu_ref.cpp -- validate the ORACLE.
//
// Everything else in this project is checked against src/cpu_ref/*.cpp, so the
// oracle itself has to be checked against something independent. That something is
// tools/gen_reference.py (PyTorch, or NumPy as a fallback), which computes each
// operation in float64 and writes inputs plus expected outputs to tests/golden/.
//
// This file therefore does two jobs:
//
//   1. Sanity-check the reference DATA. If the generator is wrong, every downstream
//      test is confidently wrong in the same direction. Softmax rows must sum to 1,
//      no golden file may contain NaN, and so on. These properties are checked
//      directly rather than trusted.
//   2. Compare the CPU oracle against that data, element by element.
//
// If reference data has not been generated the tests SKIP, with the command to run
// printed in the summary. They do not fail -- a missing prerequisite is not a bug.
//
//   python3 tools/gen_reference.py
//===----------------------------------------------------------------------===//

#include "golden.hpp"
#include "test_framework.hpp"

#include <engine/cpu_ref.hpp>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// TOLERANCES, and why each one is what it is.
//
// The reference is computed in float64 and stored as float32. The oracle computes
// in float32 with float64 accumulators. So the expected disagreement is on the
// order of one float32 ulp (relative 1.19e-7) per rounding step involved.
//===----------------------------------------------------------------------===//

/// A single float add rounds once whether it is done in float or in double, so
/// vector_add must agree BIT FOR BIT. Loosening this would hide a real bug: there
/// is no legitimate source of error in `a[i] + b[i]`.
constexpr double kExactRtol = 0.0;
constexpr double kExactAtol = 0.0;

/// One float32 rounding of the stored reference, plus one in the oracle's output.
/// Used for reductions and normalisations, whose interiors are float64 on both sides.
constexpr double kFloatRtol = 1e-6;
constexpr double kFloatAtol = 1e-9;

/// Softmax needs a looser bound than the other float ops, and the reason is worth
/// understanding because it recurs in the CUDA kernel and again in FlashAttention.
///
/// The oracle computes `std::exp(src[c] - m)` in float. glibc's exp is accurate to
/// well under 1 ulp, so exp itself is not the problem -- the ARGUMENT is. The float
/// subtraction (x - m) has an absolute error of up to half an ulp of its own
/// magnitude, i.e. 2^-24 * |x - m|. And since d(exp(x))/exp(x) = dx, that absolute
/// argument error becomes a RELATIVE output error of the same size:
///
///     rel_err  ~=  2^-24 * |x - m|  ~=  6e-8 * |x - m|
///
/// So accuracy degrades linearly with how far an element sits below the row maximum.
/// These cases have shifts up to ~40, predicting ~2.4e-6 -- and the measured worst
/// case across the whole suite is 1.15e-6, which matches. 1e-5 leaves ~4x headroom.
///
/// Two consequences for later:
///   * The CUDA kernel must not be held to a tighter bound than this one.
///   * If you use __expf() (the fast hardware approximation, ~2 ulp) instead of
///     expf(), loosen this again -- and say so in the report rather than quietly
///     widening the tolerance.
///
/// atol stays ~zero on purpose: the interesting outputs are tiny (exp(-40) ~ 4e-18)
/// and must still be right in a relative sense. An atol big enough to cover them
/// would make the test pass on an all-zeros output.
constexpr double kSoftmaxRtol = 1e-5;
constexpr double kSoftmaxAtol = 1e-30;

/// matmul sums K products. Both sides accumulate in float64, but in different
/// orders (the oracle sequentially, NumPy/BLAS in blocks), so cancellation-heavy
/// output elements can differ by more than one ulp *relative to their own small
/// magnitude*. atol is set relative to the scale of the inputs, which the generator
/// fixes at O(1/sqrt(K)); a genuine matmul bug produces O(1) relative errors and is
/// nowhere near this tolerance.
constexpr double kMatmulRtol = 1e-5;
constexpr double kMatmulAtol = 1e-6;

//===----------------------------------------------------------------------===//
// Shape tables. These mirror tools/gen_reference.py, and the stem is derived from
// the shape by the same rule on both sides, so the two cannot drift apart silently:
// if they do, the file is not found and the test SKIPS with the stem it wanted.
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
const RmsnormLinearShape kRmsnormLinearShapes[] = {
    {1, 4096, 4096, true},
    {32, 4096, 4096, true},
    {1, 12288, 4096, true},
    {17, 127, 31, true},
    {32, 4096, 4096, false}};

/// The eps the generator used. Must match, or rmsnorm disagrees at eps scale on
/// near-zero rows (which is exactly what the rmsnorm__zeros case tests).
constexpr float kEps = 1e-5f;

std::string dims2(std::int64_t a, std::int64_t b) {
  return std::to_string(a) + "x" + std::to_string(b);
}

}  // namespace

// LOAD_GOLDEN lives in golden.hpp and CHECK_CASE in test_framework.hpp -- both are
// used by tests/test_kernels.cu as well, and a macro defined twice with slightly
// different bodies is a bug waiting to happen.

//===----------------------------------------------------------------------===//
// PART 1 -- is the reference data itself trustworthy?
//===----------------------------------------------------------------------===//

TEST(golden, every_file_in_the_directory_parses) {
  // Walks tests/golden/ rather than using a hard-coded list, so a case added to
  // gen_reference.py is validated here automatically. Also proves the writer and
  // the reader agree about the format for every file, not just the ones used below.
  std::error_code ec;
  const std::filesystem::path dir(engtest::golden_dir());
  if (!std::filesystem::is_directory(dir, ec)) {
    SKIP_TEST(engtest::golden_missing_hint());
  }

  std::size_t files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (entry.path().extension() != ".bin") continue;
    ++files;
    const std::string stem = entry.path().stem().string();

    engtest::GoldenFile g;
    std::string err;
    if (!engtest::GoldenFile::try_load(stem, g, err)) {
      ctx.add_failure(__FILE__, __LINE__, "failed to parse " + stem + ": " + err);
      continue;
    }

    EXPECT_TRUE(g.size() >= 2);          // at least one input and one expected
    EXPECT_TRUE(g.has("expected"));      // the naming convention, enforced

    // No golden file may contain a NaN or an infinity. If one does, the generator
    // produced garbage and every test that consumes it is meaningless.
    for (const auto& name : g.names()) {
      const auto& t = g.at(name);
      std::size_t bad = 0;
      for (float v : t.data) {
        if (!std::isfinite(v)) ++bad;
      }
      if (bad != 0) {
        ctx.add_failure(__FILE__, __LINE__,
                        stem + "/" + name + ": " + std::to_string(bad) +
                            " non-finite value(s) in the REFERENCE data");
      }
    }
  }

  EXPECT_TRUE(files > 0);
}

TEST(golden, softmax_reference_is_a_probability_distribution) {
  // Properties that must hold for any correct softmax, checked on the reference
  // itself. This is the check that would catch a generator that, say, normalised
  // along the wrong axis -- a mistake the oracle comparison alone would not reveal,
  // because a matching bug in both is still a match.
  for (const auto& s : kSoftmaxShapes) {
    const std::string stem = "softmax_rows__" + dims2(s.rows, s.cols);
    LOAD_GOLDEN(g, stem);
    const auto& exp = g.at("expected");
    ASSERT_EQ(exp.numel(), s.rows * s.cols);

    for (std::int64_t r = 0; r < s.rows; ++r) {
      double sum = 0.0;
      for (std::int64_t c = 0; c < s.cols; ++c) {
        const float v = exp.data[static_cast<std::size_t>(r * s.cols + c)];
        if (v < 0.0f || v > 1.0f) {
          ctx.add_failure(__FILE__, __LINE__,
                          stem + ": probability out of [0,1] at (" +
                              std::to_string(r) + "," + std::to_string(c) +
                              ") = " + std::to_string(v));
          break;
        }
        sum += static_cast<double>(v);
      }
      // Row sums of float32 probabilities: the error accumulates over `cols` adds,
      // so the tolerance has to scale with the width. 50257 columns cannot sum to
      // 1.0 within 1e-6.
      const double tol = 4e-7 * static_cast<double>(s.cols) + 1e-6;
      EXPECT_NEAR(sum, 1.0, tol);
    }
  }
}

TEST(golden, softmax_edge_cases_are_what_they_claim_to_be) {
  LOAD_GOLDEN(g, "softmax_rows__edge");
  const auto& in = g.at("input");
  const auto& exp = g.at("expected");
  ASSERT_EQ(in.numel(), std::int64_t{4 * 512});
  ASSERT_EQ(exp.numel(), std::int64_t{4 * 512});

  // Row 1 contains +300. exp(300) is +inf in float32 and in float64, so if the
  // generator had skipped the max subtraction this row would be all NaN. Confirming
  // it is finite confirms the reference is stable, which is the whole premise.
  for (std::size_t c = 0; c < 512; ++c) {
    EXPECT_TRUE(std::isfinite(exp.data[512 + c]));
  }
  EXPECT_NEAR(exp.data[512 + 100], 1.0, 1e-6);  // the +300 element takes ~all mass

  // Row 2 is constant, so every output must be exactly 1/512 -- the one case where
  // softmax has a closed form worth asserting.
  for (std::size_t c = 0; c < 512; ++c) {
    EXPECT_NEAR(exp.data[2 * 512 + c], 1.0 / 512.0, 1e-9);
  }

  // Row 3 is one +20 among -20s: a near one-hot with tiny but NONZERO tails.
  // If the tails were zero, float32 had underflowed and the case would not be
  // exercising relative accuracy at all.
  EXPECT_NEAR(exp.data[3 * 512 + 7], 1.0, 1e-6);
  EXPECT_TRUE(exp.data[3 * 512 + 0] > 0.0f);
  EXPECT_TRUE(exp.data[3 * 512 + 0] < 1e-16f);
}

TEST(golden, rmsnorm_of_zeros_is_zeros_not_nan) {
  // The division is x / sqrt(mean(x^2) + eps). With an all-zero row that is
  // 0 / sqrt(eps), which is 0 -- but only because eps is inside the sqrt. Drop eps
  // and it becomes 0/0 = NaN. This case pins the convention down in data.
  LOAD_GOLDEN(g, "rmsnorm__zeros");
  const auto& exp = g.at("expected");
  ASSERT_EQ(exp.numel(), std::int64_t{2 * 256});
  for (float v : exp.data) {
    EXPECT_EQ(v, 0.0f);
  }
}

TEST(golden, missing_file_reports_the_fix) {
  engtest::GoldenFile g;
  std::string err;
  EXPECT_FALSE(engtest::GoldenFile::try_load("this_stem_does_not_exist", g, err));
  // The error message must contain the command to run. A skip that does not tell you
  // what to do is only marginally better than a silent pass.
  EXPECT_TRUE(err.find("gen_reference.py") != std::string::npos);
  EXPECT_THROWS(engtest::GoldenFile::load("this_stem_does_not_exist"));
}

TEST(golden, unknown_tensor_name_throws_and_lists_alternatives) {
  LOAD_GOLDEN(g, "matmul__1x1x1");
  EXPECT_TRUE(g.has("A"));
  EXPECT_TRUE(g.has("B"));
  EXPECT_TRUE(g.has("expected"));
  EXPECT_FALSE(g.has("C"));
  EXPECT_THROWS(g.at("C"));  // a typo in a tensor name must never silently pass

  const auto& a = g.at("A");
  ASSERT_EQ(a.numel(), std::int64_t{1});
  EXPECT_EQ(a.dim(0), std::int64_t{1});
  EXPECT_THROWS(a.dim(5));
}

//===----------------------------------------------------------------------===//
// PART 2 -- the oracle versus the reference.
//===----------------------------------------------------------------------===//

TEST(cpu_ref, vector_add_is_bit_exact) {
  for (std::int64_t n : kSizes1D) {
    const std::string stem = "vector_add__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);
    const auto& a = g.at("a");
    const auto& b = g.at("b");
    const auto& exp = g.at("expected");
    ASSERT_EQ(a.numel(), n);
    ASSERT_EQ(b.numel(), n);
    ASSERT_EQ(exp.numel(), n);

    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);
    engine::cpu::vector_add(a.data.data(), b.data.data(), out.data(), n);

    CHECK_CASE(stem, out.data(), exp.data.data(), out.size(), kExactRtol, kExactAtol);
  }
}

TEST(cpu_ref, reduce_sum_matches_float64_reference) {
  for (std::int64_t n : kSizes1D) {
    const std::string stem = "reduce_sum__n" + std::to_string(n);
    LOAD_GOLDEN(g, stem);
    const auto& x = g.at("x");
    const auto& exp = g.at("expected");
    ASSERT_EQ(x.numel(), n);
    ASSERT_EQ(exp.numel(), std::int64_t{1});

    const double got = engine::cpu::reduce_sum(x.data.data(), n);

    // The reference is stored as float32, so the comparison is only meaningful to
    // float32 precision -- cast the oracle's double down before comparing rather
    // than pretending the stored value has more digits than it does.
    const float got_f = static_cast<float>(got);
    CHECK_CASE(stem, &got_f, exp.data.data(), std::size_t{1}, kFloatRtol, kFloatAtol);
  }
}

TEST(cpu_ref, reduce_sum_of_empty_is_zero) {
  // Not in the golden set because the format cannot express a zero-length input
  // usefully; asserted here because the CUDA kernel must match this and n == 0 is a
  // real case (an empty batch).
  EXPECT_EQ(engine::cpu::reduce_sum(nullptr, 0), 0.0);
}

TEST(cpu_ref, softmax_rows_matches_reference) {
  for (const auto& s : kSoftmaxShapes) {
    const std::string stem = "softmax_rows__" + dims2(s.rows, s.cols);
    LOAD_GOLDEN(g, stem);
    const auto& in = g.at("input");
    const auto& exp = g.at("expected");
    ASSERT_EQ(in.numel(), s.rows * s.cols);

    std::vector<float> out(static_cast<std::size_t>(s.rows * s.cols), 0.0f);
    engine::cpu::softmax_rows(in.data.data(), out.data(), s.rows, s.cols);

    CHECK_CASE(stem, out.data(), exp.data.data(), out.size(), kSoftmaxRtol,
               kSoftmaxAtol);
  }
}

TEST(cpu_ref, softmax_rows_survives_the_edge_cases) {
  LOAD_GOLDEN(g, "softmax_rows__edge");
  const auto& in = g.at("input");
  const auto& exp = g.at("expected");

  std::vector<float> out(static_cast<std::size_t>(4 * 512), 0.0f);
  engine::cpu::softmax_rows(in.data.data(), out.data(), 4, 512);

  CHECK_CASE("softmax_rows__edge", out.data(), exp.data.data(), out.size(),
             kSoftmaxRtol, kSoftmaxAtol);

  // Independently of the reference: the output must be finite everywhere. The +300
  // row is precisely the input that turns a naive softmax into NaN.
  for (float v : out) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(cpu_ref, rmsnorm_matches_reference) {
  for (const auto& s : kRmsShapes) {
    const std::string stem =
        "rmsnorm__" + dims2(s.rows, s.cols) + (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);
    const auto& in = g.at("input");
    const auto& exp = g.at("expected");
    ASSERT_EQ(in.numel(), s.rows * s.cols);

    const float* weight = nullptr;
    if (s.with_weight) {
      ASSERT_TRUE(g.has("weight"));
      ASSERT_EQ(g.at("weight").numel(), s.cols);
      weight = g.at("weight").data.data();
    } else {
      EXPECT_FALSE(g.has("weight"));  // the nullptr path really is being exercised
    }

    std::vector<float> out(static_cast<std::size_t>(s.rows * s.cols), 0.0f);
    engine::cpu::rmsnorm(in.data.data(), weight, out.data(), s.rows, s.cols, kEps);

    CHECK_CASE(stem, out.data(), exp.data.data(), out.size(), kFloatRtol, kFloatAtol);
  }
}

TEST(cpu_ref, rmsnorm_of_zeros_is_zeros) {
  LOAD_GOLDEN(g, "rmsnorm__zeros");
  const auto& in = g.at("input");
  const auto& exp = g.at("expected");

  std::vector<float> out(static_cast<std::size_t>(2 * 256), 1.0f);  // pre-fill != 0
  engine::cpu::rmsnorm(in.data.data(), nullptr, out.data(), 2, 256, kEps);

  CHECK_CASE("rmsnorm__zeros", out.data(), exp.data.data(), out.size(), kFloatRtol,
             kFloatAtol);
  for (float v : out) {
    EXPECT_FALSE(std::isnan(v));
  }
}

TEST(cpu_ref, rmsnorm_null_weight_equals_unit_weight) {
  // An invariant rather than a data comparison: passing nullptr must be identical to
  // passing an all-ones gain. Worth pinning because the CUDA kernel will branch on
  // `weight == nullptr` and it is easy to get that branch subtly wrong.
  LOAD_GOLDEN(g, "rmsnorm__32x4096_now");
  const auto& in = g.at("input");
  const std::int64_t rows = 32, cols = 4096;

  std::vector<float> ones(static_cast<std::size_t>(cols), 1.0f);
  std::vector<float> a(static_cast<std::size_t>(rows * cols), 0.0f);
  std::vector<float> b(static_cast<std::size_t>(rows * cols), 0.0f);

  engine::cpu::rmsnorm(in.data.data(), nullptr, a.data(), rows, cols, kEps);
  engine::cpu::rmsnorm(in.data.data(), ones.data(), b.data(), rows, cols, kEps);

  // Multiplying by exactly 1.0f is exact in IEEE-754, so this is bit-for-bit.
  CHECK_CASE("rmsnorm null vs ones", a.data(), b.data(), a.size(), kExactRtol,
             kExactAtol);
}

TEST(cpu_ref, matmul_matches_reference) {
  for (const auto& s : kMatmulShapes) {
    const std::string stem = "matmul__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k);
    LOAD_GOLDEN(g, stem);
    const auto& A = g.at("A");
    const auto& B = g.at("B");
    const auto& exp = g.at("expected");

    // Shape assertions first. A transposed golden file would otherwise show up as a
    // baffling numerical failure instead of an obvious shape mismatch.
    ASSERT_EQ(A.numel(), s.m * s.k);
    ASSERT_EQ(B.numel(), s.k * s.n);
    ASSERT_EQ(exp.numel(), s.m * s.n);
    ASSERT_EQ(A.dim(0), s.m);
    ASSERT_EQ(A.dim(1), s.k);
    ASSERT_EQ(B.dim(0), s.k);
    ASSERT_EQ(B.dim(1), s.n);

    std::vector<float> C(static_cast<std::size_t>(s.m * s.n), 0.0f);
    engine::cpu::matmul(A.data.data(), B.data.data(), C.data(), s.m, s.n, s.k);

    CHECK_CASE(stem, C.data(), exp.data.data(), C.size(), kMatmulRtol, kMatmulAtol);
  }
}

TEST(cpu_ref, matmul_identity_is_a_copy) {
  // A property test that needs no reference data: A * I == A. It catches row/column
  // swaps that a symmetric test shape would let through, which is why the shape here
  // is deliberately non-square.
  const std::int64_t M = 5, K = 3;
  std::vector<float> A(static_cast<std::size_t>(M * K));
  for (std::size_t i = 0; i < A.size(); ++i) {
    A[i] = static_cast<float>(i) * 0.5f - 3.0f;
  }

  std::vector<float> I(static_cast<std::size_t>(K * K), 0.0f);
  for (std::int64_t d = 0; d < K; ++d) {
    I[static_cast<std::size_t>(d * K + d)] = 1.0f;
  }

  std::vector<float> C(static_cast<std::size_t>(M * K), -1.0f);
  engine::cpu::matmul(A.data(), I.data(), C.data(), M, /*N=*/K, K);

  EXPECT_ALLCLOSE(C.data(), A.data(), C.size(), kExactRtol, kExactAtol);
}

//===----------------------------------------------------------------------===//
// Module 3 -- Fused operations tests
//===----------------------------------------------------------------------===//

TEST(cpu_ref, residual_rmsnorm_matches_reference) {
  for (const auto& s : kResidualRmsShapes) {
    const std::string stem =
        "residual_rmsnorm__" + dims2(s.rows, s.cols) + (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);
    const auto& x = g.at("x");
    const auto& res = g.at("residual");
    const auto& exp_norm = g.at("expected");
    const auto& exp_sum = g.at("sum_out");
    ASSERT_EQ(x.numel(), s.rows * s.cols);
    ASSERT_EQ(res.numel(), s.rows * s.cols);
    ASSERT_EQ(exp_norm.numel(), s.rows * s.cols);
    ASSERT_EQ(exp_sum.numel(), s.rows * s.cols);

    const float* weight = nullptr;
    if (s.with_weight) {
      ASSERT_TRUE(g.has("weight"));
      ASSERT_EQ(g.at("weight").numel(), s.cols);
      weight = g.at("weight").data.data();
    } else {
      EXPECT_FALSE(g.has("weight"));
    }

    std::vector<float> norm_out(static_cast<std::size_t>(s.rows * s.cols), 0.0f);
    std::vector<float> sum_out(static_cast<std::size_t>(s.rows * s.cols), 0.0f);
    engine::cpu::residual_rmsnorm(x.data.data(), res.data.data(), weight,
                                  norm_out.data(), sum_out.data(), s.rows, s.cols, kEps);

    CHECK_CASE(stem + " (norm)", norm_out.data(), exp_norm.data.data(), norm_out.size(),
               kFloatRtol, kFloatAtol);
    CHECK_CASE(stem + " (sum)", sum_out.data(), exp_sum.data.data(), sum_out.size(),
               kFloatRtol, kFloatAtol);
  }
}

TEST(cpu_ref, residual_rmsnorm_of_zeros_is_zeros) {
  LOAD_GOLDEN(g, "residual_rmsnorm__zeros");
  const auto& x = g.at("x");
  const auto& res = g.at("residual");
  const auto& exp_norm = g.at("expected");
  const auto& exp_sum = g.at("sum_out");

  std::vector<float> norm_out(static_cast<std::size_t>(2 * 256), 1.0f);
  std::vector<float> sum_out(static_cast<std::size_t>(2 * 256), 1.0f);
  engine::cpu::residual_rmsnorm(x.data.data(), res.data.data(), nullptr,
                                norm_out.data(), sum_out.data(), 2, 256, kEps);

  CHECK_CASE("residual_rmsnorm__zeros (norm)", norm_out.data(), exp_norm.data.data(),
             norm_out.size(), kFloatRtol, kFloatAtol);
  CHECK_CASE("residual_rmsnorm__zeros (sum)", sum_out.data(), exp_sum.data.data(),
             sum_out.size(), kFloatRtol, kFloatAtol);
  for (float v : norm_out) {
    EXPECT_FALSE(std::isnan(v));
  }
}

TEST(cpu_ref, residual_rmsnorm_null_weight_equals_unit_weight) {
  LOAD_GOLDEN(g, "residual_rmsnorm__32x4096_now");
  const auto& x = g.at("x");
  const auto& res = g.at("residual");
  const std::int64_t rows = 32, cols = 4096;

  std::vector<float> ones(static_cast<std::size_t>(cols), 1.0f);
  std::vector<float> norm_a(static_cast<std::size_t>(rows * cols), 0.0f);
  std::vector<float> sum_a(static_cast<std::size_t>(rows * cols), 0.0f);
  std::vector<float> norm_b(static_cast<std::size_t>(rows * cols), 0.0f);
  std::vector<float> sum_b(static_cast<std::size_t>(rows * cols), 0.0f);

  engine::cpu::residual_rmsnorm(x.data.data(), res.data.data(), nullptr,
                                norm_a.data(), sum_a.data(), rows, cols, kEps);
  engine::cpu::residual_rmsnorm(x.data.data(), res.data.data(), ones.data(),
                                norm_b.data(), sum_b.data(), rows, cols, kEps);

  CHECK_CASE("residual_rmsnorm null vs ones (norm)", norm_a.data(), norm_b.data(),
             norm_a.size(), kExactRtol, kExactAtol);
  CHECK_CASE("residual_rmsnorm null vs ones (sum)", sum_a.data(), sum_b.data(),
             sum_a.size(), kExactRtol, kExactAtol);
}

TEST(cpu_ref, rmsnorm_linear_matches_reference) {
  for (const auto& s : kRmsnormLinearShapes) {
    const std::string stem = "rmsnorm_linear__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k) +
                             (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);
    const auto& in = g.at("input");
    const auto& W = g.at("W");
    const auto& exp = g.at("expected");

    ASSERT_EQ(in.numel(), s.m * s.k);
    ASSERT_EQ(W.numel(), s.k * s.n);
    ASSERT_EQ(exp.numel(), s.m * s.n);

    const float* weight = nullptr;
    if (s.with_weight) {
      ASSERT_TRUE(g.has("weight"));
      ASSERT_EQ(g.at("weight").numel(), s.k);
      weight = g.at("weight").data.data();
    } else {
      EXPECT_FALSE(g.has("weight"));
    }

    std::vector<float> out(static_cast<std::size_t>(s.m * s.n), 0.0f);
    engine::cpu::rmsnorm_linear(in.data.data(), weight, W.data.data(), out.data(),
                                s.m, s.n, s.k, kEps);

    CHECK_CASE(stem, out.data(), exp.data.data(), out.size(), kMatmulRtol, kMatmulAtol);
  }
}

TEST(cpu_ref, rmsnorm_linear_null_weight_equals_unit_weight) {
  LOAD_GOLDEN(g, "rmsnorm_linear__32x4096x4096_now");
  const auto& in = g.at("input");
  const auto& W = g.at("W");
  const std::int64_t M = 32, N = 4096, K = 4096;

  std::vector<float> ones(static_cast<std::size_t>(K), 1.0f);
  std::vector<float> out_a(static_cast<std::size_t>(M * N), 0.0f);
  std::vector<float> out_b(static_cast<std::size_t>(M * N), 0.0f);

  engine::cpu::rmsnorm_linear(in.data.data(), nullptr, W.data.data(), out_a.data(),
                              M, N, K, kEps);
  engine::cpu::rmsnorm_linear(in.data.data(), ones.data(), W.data.data(), out_b.data(),
                              M, N, K, kEps);

  CHECK_CASE("rmsnorm_linear null vs ones", out_a.data(), out_b.data(), out_a.size(),
             kExactRtol, kExactAtol);
}

TEST(cpu_ref, rmsnorm_linear_agrees_with_separate_rmsnorm_and_matmul) {
  for (const auto& s : kRmsnormLinearShapes) {
    const std::string stem = "rmsnorm_linear__" + std::to_string(s.m) + "x" +
                             std::to_string(s.n) + "x" + std::to_string(s.k) +
                             (s.with_weight ? "_w" : "_now");
    LOAD_GOLDEN(g, stem);
    const auto& in = g.at("input");
    const auto& W = g.at("W");

    const float* weight = s.with_weight ? g.at("weight").data.data() : nullptr;

    std::vector<float> fused_out(static_cast<std::size_t>(s.m * s.n), 0.0f);
    engine::cpu::rmsnorm_linear(in.data.data(), weight, W.data.data(), fused_out.data(),
                                s.m, s.n, s.k, kEps);

    std::vector<float> temp(static_cast<std::size_t>(s.m * s.k), 0.0f);
    std::vector<float> separate_out(static_cast<std::size_t>(s.m * s.n), 0.0f);
    engine::cpu::rmsnorm(in.data.data(), weight, temp.data(), s.m, s.k, kEps);
    engine::cpu::matmul(temp.data(), W.data.data(), separate_out.data(), s.m, s.n, s.k);

    CHECK_CASE(stem + " (fused vs separate)", fused_out.data(), separate_out.data(),
               fused_out.size(), kExactRtol, kExactAtol);
  }
}
