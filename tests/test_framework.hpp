#pragma once
//===----------------------------------------------------------------------===//
// tests/test_framework.hpp -- a ~200 line test framework.
//
// WHY NOT GOOGLETEST?
//
// Three reasons, in order of weight:
//
//   1. No network dependency. GoogleTest via FetchContent needs to download and
//      compile on first configure. This project is built on two separate machines
//      and in CI; a self-contained harness always works.
//   2. Float comparison is the ENTIRE JOB here. Comparing a CUDA kernel against a
//      CPU oracle needs an assertion that says "element 8417 of 65536 differs:
//      got 0.4213, expected 0.4198, rel err 3.6e-3" -- GoogleTest's EXPECT_NEAR on
//      a loop just says which iteration failed and stops. EXPECT_ALLCLOSE below is
//      built for the actual task.
//   3. It compiles in under a second, so the test-edit-test loop stays tight.
//
// If you later want GoogleTest's fixtures or death tests, adding it is easy. Until
// then this is less code than the integration would be.
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace engtest {

//===----------------------------------------------------------------------===//
// describe(): stringify a value for failure messages.
// Overloads are found by ordinary overload resolution from inside this namespace,
// which is why the vector overloads must be declared before use below.
//===----------------------------------------------------------------------===//
template <typename T>
std::string describe(const T& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

inline std::string describe(bool v) { return v ? "true" : "false"; }

template <typename T>
std::string describe(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i != 0) oss << ',';
    oss << v[i];
  }
  oss << ']';
  return oss.str();
}

//===----------------------------------------------------------------------===//
// Per-test failure collection. EXPECT_* records and continues; ASSERT_* returns.
//===----------------------------------------------------------------------===//
struct Failure {
  std::string file;
  int line = 0;
  std::string message;
};

class TestContext {
 public:
  void add_failure(const char* file, int line, const std::string& message) {
    failures_.push_back({file, line, message});
  }
  bool ok() const { return failures_.empty(); }
  const std::vector<Failure>& failures() const { return failures_; }

  /// Marks the test as not-applicable in this environment rather than failed.
  /// Used when a prerequisite is genuinely absent: no CUDA device, or golden
  /// reference files that have not been generated yet. A skip is honest; a silent
  /// pass would be a lie and a hard failure would be noise.
  void skip(const std::string& reason) {
    skipped_ = true;
    skip_reason_ = reason;
  }
  bool skipped() const { return skipped_; }
  const std::string& skip_reason() const { return skip_reason_; }

 private:
  std::vector<Failure> failures_;
  bool skipped_ = false;
  std::string skip_reason_;
};

using TestFn = void (*)(TestContext&);

struct TestCase {
  std::string suite;
  std::string name;
  TestFn fn = nullptr;
  bool pending = false;
  std::string file;
};

std::vector<TestCase>& registry();

/// Static-initialiser hook: constructing one of these at namespace scope registers
/// a test before main() runs.
struct Registrar {
  Registrar(const char* suite, const char* name, TestFn fn, bool pending,
            const char* file);
};

/// Entry point. Flags: --filter=SUBSTR, --list, --verbose.
int run_all(int argc, char** argv);

//===----------------------------------------------------------------------===//
// Assertion implementations
//===----------------------------------------------------------------------===//
void report_bool(TestContext& ctx, bool cond, bool expected, const char* expr,
                 const char* file, int line);

template <typename A, typename B>
void report_eq(TestContext& ctx, const A& a, const B& b, const char* ea, const char* eb,
               const char* file, int line) {
  if (!(a == b)) {
    std::ostringstream oss;
    oss << "expected " << ea << " == " << eb << "\n"
        << "    actual:   " << describe(a) << "\n"
        << "    expected: " << describe(b);
    ctx.add_failure(file, line, oss.str());
  }
}

void report_near(TestContext& ctx, double a, double b, double tol, const char* ea,
                 const char* eb, const char* file, int line);

/// Element-wise comparison of two float arrays, numpy semantics:
///     pass if |actual - expected| <= atol + rtol * |expected|
///
/// Reports the total mismatch count and the single WORST element by relative error,
/// which is the information you actually need when a kernel is subtly wrong. A
/// handful of mismatches usually means an edge/bounds bug; a majority mismatching
/// usually means an indexing or layout bug. The distinction saves a lot of time.
///
/// Returns true on pass.
bool report_allclose(TestContext& ctx, const float* actual, const float* expected,
                     std::size_t n, double rtol, double atol, const char* label,
                     const char* file, int line);

/// True if `what` looks like one of the project's "not implemented yet" stub
/// exceptions (src/tensor.cpp throws these, and so do the kernel launchers).
///
/// WHY THE FRAMEWORK KNOWS ABOUT THIS: a negative test such as
/// `EXPECT_THROWS(t.reshape({5, 5}))` is satisfied by ANY exception -- including the
/// stub's own "not implemented". So while the module is unwritten, every negative
/// test passes vacuously, and the runner cheerfully reports them as ready to
/// promote. That is precisely the sort of green-but-meaningless test this harness
/// exists to avoid, so EXPECT_THROWS rejects the sentinel explicitly.
bool is_unimplemented_error(const char* what);

}  // namespace engtest

//===----------------------------------------------------------------------===//
// Registration macros
//===----------------------------------------------------------------------===//
#define ENGTEST_REGISTER(suite, name, pending)                                     \
  static void engtest_##suite##_##name(::engtest::TestContext&);                   \
  static ::engtest::Registrar engtest_reg_##suite##_##name(                        \
      #suite, #name, &engtest_##suite##_##name, (pending), __FILE__);              \
  static void engtest_##suite##_##name(                                            \
      [[maybe_unused]] ::engtest::TestContext& ctx)

/// A normal test. Must pass.
#define TEST(suite, name) ENGTEST_REGISTER(suite, name, false)

/// A test for functionality that is not implemented yet. It still runs, but a
/// failure is reported as PENDING rather than FAILED, so the suite stays green
/// while the module is outstanding. If it unexpectedly passes, the runner tells you
/// to promote it to TEST.
///
/// This is how the Module 1 tensor tests and the CUDA kernel exercises are marked.
/// Promote each one as you implement it -- the pending count is your progress bar.
#define TEST_PENDING(suite, name) ENGTEST_REGISTER(suite, name, true)

//===----------------------------------------------------------------------===//
// Assertion macros. EXPECT_* continues on failure; ASSERT_* aborts the test.
//===----------------------------------------------------------------------===//
#define EXPECT_TRUE(cond) \
  ::engtest::report_bool(ctx, static_cast<bool>(cond), true, #cond, __FILE__, __LINE__)

#define EXPECT_FALSE(cond) \
  ::engtest::report_bool(ctx, static_cast<bool>(cond), false, #cond, __FILE__, __LINE__)

#define EXPECT_EQ(a, b) ::engtest::report_eq(ctx, (a), (b), #a, #b, __FILE__, __LINE__)

#define EXPECT_NEAR(a, b, tol)                                                     \
  ::engtest::report_near(ctx, static_cast<double>(a), static_cast<double>(b),       \
                         static_cast<double>(tol), #a, #b, __FILE__, __LINE__)

/// The workhorse for kernel verification.
#define EXPECT_ALLCLOSE(actual, expected, n, rtol, atol)                           \
  ::engtest::report_allclose(ctx, (actual), (expected), (n), (rtol), (atol),        \
                             #actual " vs " #expected, __FILE__, __LINE__)

/// EXPECT_ALLCLOSE inside a loop over test cases, plus the name of the case that
/// failed.
///
/// Without this you get "allclose failed for host.data() vs expected.data()" and no
/// indication of WHICH shape produced it -- and with 9 shapes per operation, the
/// difference between "matmul is broken" and "matmul is broken only at M=17, K=23"
/// is the difference between an afternoon and five minutes. The second line is
/// appended as a separate failure so it appears directly beneath the numbers.
#define CHECK_CASE(stem, actual, expected, n, rtol, atol)                          \
  do {                                                                             \
    if (!EXPECT_ALLCLOSE((actual), (expected), (n), (rtol), (atol))) {              \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      "  ^^ failing case: " + std::string(stem));                  \
    }                                                                              \
  } while (0)

#define ASSERT_TRUE(cond)                                                          \
  do {                                                                             \
    if (!static_cast<bool>(cond)) {                                                \
      ctx.add_failure(__FILE__, __LINE__, "ASSERT_TRUE failed: " #cond);           \
      return;                                                                      \
    }                                                                              \
  } while (0)

#define ASSERT_EQ(a, b)                                                            \
  do {                                                                             \
    if (!((a) == (b))) {                                                           \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      std::string("ASSERT_EQ failed: " #a " == " #b "\n") +        \
                          "    actual:   " + ::engtest::describe(a) + "\n" +       \
                          "    expected: " + ::engtest::describe(b));              \
      return;                                                                      \
    }                                                                              \
  } while (0)

/// Asserts that `stmt` throws SOMETHING OTHER than a "not implemented" stub error.
/// Used heavily on the Tensor API, where rejecting bad input with a clear error IS
/// the specified behaviour. See is_unimplemented_error() for why the exclusion
/// matters: without it these tests pass before a single line is written.
#define EXPECT_THROWS(stmt)                                                        \
  do {                                                                             \
    bool engtest_threw = false;                                                    \
    bool engtest_stub = false;                                                     \
    try {                                                                          \
      stmt;                                                                        \
    } catch (const std::exception& engtest_e) {                                    \
      engtest_threw = true;                                                        \
      engtest_stub = ::engtest::is_unimplemented_error(engtest_e.what());          \
    } catch (...) {                                                                \
      engtest_threw = true;                                                        \
    }                                                                              \
    if (!engtest_threw) {                                                          \
      ctx.add_failure(__FILE__, __LINE__, "expected an exception from: " #stmt);   \
    } else if (engtest_stub) {                                                     \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      "still a stub, so this proves nothing yet: " #stmt);         \
    }                                                                              \
  } while (0)

/// Like EXPECT_THROWS, but also requires the message to contain `substr`. Use it
/// when the wording of the error is part of the contract -- "insert .contiguous()"
/// is advice the caller needs, not decoration.
#define EXPECT_THROWS_MSG(stmt, substr)                                            \
  do {                                                                             \
    bool engtest_threw = false;                                                    \
    std::string engtest_msg;                                                       \
    try {                                                                          \
      stmt;                                                                        \
    } catch (const std::exception& engtest_e) {                                    \
      engtest_threw = true;                                                        \
      engtest_msg = engtest_e.what();                                              \
    } catch (...) {                                                                \
      engtest_threw = true;                                                        \
      engtest_msg = "<non-std exception>";                                         \
    }                                                                              \
    if (!engtest_threw) {                                                          \
      ctx.add_failure(__FILE__, __LINE__, "expected an exception from: " #stmt);   \
    } else if (::engtest::is_unimplemented_error(engtest_msg.c_str())) {           \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      "still a stub, so this proves nothing yet: " #stmt);         \
    } else if (engtest_msg.find(substr) == std::string::npos) {                    \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      std::string("exception from " #stmt                          \
                                  " should mention \"" substr "\"\n    got: ") +  \
                          engtest_msg);                                            \
    }                                                                              \
  } while (0)

#define EXPECT_NO_THROW(stmt)                                                      \
  do {                                                                             \
    try {                                                                          \
      stmt;                                                                        \
    } catch (const std::exception& e) {                                            \
      ctx.add_failure(__FILE__, __LINE__,                                          \
                      std::string("unexpected exception from " #stmt ": ") +       \
                          e.what());                                               \
    }                                                                              \
  } while (0)

/// Abandons the test as not-applicable. Use for missing prerequisites only --
/// never to dodge a real failure.
#define SKIP_TEST(reason)   \
  do {                      \
    ctx.skip(reason);       \
    return;                 \
  } while (0)
