#include "test_framework.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>

namespace engtest {

std::vector<TestCase>& registry() {
  // Function-local static: guarantees the vector is constructed before the first
  // Registrar touches it, regardless of static initialisation order across
  // translation units. A namespace-scope global would not.
  static std::vector<TestCase> r;
  return r;
}

Registrar::Registrar(const char* suite, const char* name, TestFn fn, bool pending,
                     const char* file) {
  registry().push_back({suite, name, fn, pending, file});
}

void report_bool(TestContext& ctx, bool cond, bool expected, const char* expr,
                 const char* file, int line) {
  if (cond != expected) {
    std::ostringstream oss;
    oss << "expected " << (expected ? "true" : "false") << " from: " << expr;
    ctx.add_failure(file, line, oss.str());
  }
}

void report_near(TestContext& ctx, double a, double b, double tol, const char* ea,
                 const char* eb, const char* file, int line) {
  const double diff = std::abs(a - b);
  if (!(diff <= tol)) {  // written this way so NaN fails rather than passes
    std::ostringstream oss;
    oss << "expected " << ea << " ~= " << eb << " (tol " << tol << ")\n"
        << "    actual:   " << a << "\n"
        << "    expected: " << b << "\n"
        << "    abs diff: " << diff;
    ctx.add_failure(file, line, oss.str());
  }
}

bool report_allclose(TestContext& ctx, const float* actual, const float* expected,
                     std::size_t n, double rtol, double atol, const char* label,
                     const char* file, int line) {
  if (actual == nullptr || expected == nullptr) {
    ctx.add_failure(file, line, std::string("allclose: null pointer for ") + label);
    return false;
  }

  std::size_t mismatches = 0;
  std::size_t first_bad = 0;
  bool have_first = false;

  // Track the worst element by RELATIVE error, which is the scale-independent
  // measure. A large absolute error on a large value is usually fine; a large
  // relative error anywhere is usually a bug.
  double worst_rel = -1.0;
  std::size_t worst_idx = 0;
  std::size_t nan_count = 0;

  for (std::size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = static_cast<double>(expected[i]);

    const bool a_nan = std::isnan(a);
    const bool e_nan = std::isnan(e);

    if (a_nan || e_nan) {
      // Both NaN counts as agreement -- masked attention rows legitimately produce
      // NaN and we do not want the oracle and the kernel to disagree about that.
      if (a_nan && e_nan) continue;
      ++mismatches;
      ++nan_count;
      if (!have_first) {
        first_bad = i;
        have_first = true;
      }
      continue;
    }

    const double diff = std::abs(a - e);
    const double tol = atol + rtol * std::abs(e);
    if (diff > tol) {
      ++mismatches;
      if (!have_first) {
        first_bad = i;
        have_first = true;
      }
    }

    const double denom = std::abs(e) > 0.0 ? std::abs(e) : 1.0;
    const double rel = diff / denom;
    if (rel > worst_rel) {
      worst_rel = rel;
      worst_idx = i;
    }
  }

  if (mismatches == 0) return true;

  const double pct =
      100.0 * static_cast<double>(mismatches) / static_cast<double>(n == 0 ? 1 : n);

  std::ostringstream oss;
  oss << "allclose failed for " << label << "\n"
      << "    elements:    " << n << "\n"
      << "    mismatches:  " << mismatches << " (" << pct << "%)\n"
      << "    tolerance:   rtol=" << rtol << " atol=" << atol << "\n";

  if (nan_count > 0) {
    oss << "    NaN/inf mismatches: " << nan_count
        << "  <-- check for uninitialised memory or exp() overflow\n";
  }

  oss << "    first bad index " << first_bad << ": actual=" << actual[first_bad]
      << " expected=" << expected[first_bad] << "\n"
      << "    worst  index " << worst_idx << ": actual=" << actual[worst_idx]
      << " expected=" << expected[worst_idx] << " rel_err=" << worst_rel << "\n";

  // A diagnosis hint, because the mismatch RATIO is genuinely informative and it is
  // easy to forget to look at it.
  if (pct > 90.0) {
    oss << "    hint: nearly everything is wrong -- suspect indexing, row/column\n"
        << "          order, or a transposed layout, not an edge case.";
  } else if (pct < 5.0) {
    oss << "    hint: only a few elements are wrong -- suspect a bounds check or\n"
        << "          the last partial tile/block.";
  }

  // Separate hint, and an important one: distinguish "the code is wrong" from "the
  // tolerance is slightly too tight". If the worst relative error is within an order
  // of magnitude of rtol, this is almost certainly float rounding rather than a bug,
  // and the right fix is to justify a wider tolerance in a comment -- not to widen it
  // silently, and not to spend an afternoon hunting a bug that is not there.
  if (worst_rel > 0.0 && rtol > 0.0 && worst_rel < 10.0 * rtol) {
    oss << "\n    hint: worst rel_err (" << worst_rel << ") is within 10x of rtol ("
        << rtol << ").\n"
        << "          That smells like accumulated float rounding, not a logic error.\n"
        << "          Work out the expected error bound and record WHY the tolerance\n"
        << "          is what it is.";
  }

  ctx.add_failure(file, line, oss.str());
  return false;
}

bool is_unimplemented_error(const char* what) {
  if (what == nullptr) return false;
  // The sentinel is the phrase used by the stub helpers throughout the project:
  //   src/tensor.cpp     -> "[Module 1, not implemented] Tensor::reshape -- ..."
  //   kernels/*.cu       -> "... not implemented yet (exercise 3)"
  // Matching on the phrase rather than on an exception subclass keeps the test
  // framework independent of engine headers, which is worth more than the precision
  // an EngineError subclass would buy.
  return std::strstr(what, "not implemented") != nullptr;
}

namespace {

void print_usage() {
  std::printf(
      "usage: engine_tests [--filter=SUBSTR] [--list] [--verbose]\n"
      "  --filter=SUBSTR  run only tests whose \"suite.name\" contains SUBSTR\n"
      "  --list           list all registered tests and exit\n"
      "  --verbose        print each passing test, not just failures\n"
      "\n"
      "exit codes: 0 = all selected tests passed (pending and skipped are not\n"
      "            failures), 1 = at least one test failed, 2 = bad arguments, or a\n"
      "            --filter that matched no test at all\n");
}

}  // namespace

int run_all(int argc, char** argv) {
  std::string filter;
  bool list_only = false;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--filter=", 0) == 0) {
      filter = arg.substr(9);
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      std::printf("unknown argument: %s\n", arg.c_str());
      print_usage();
      return 2;
    }
  }

  auto& all = registry();

  // Deterministic order: sort by suite then name. Registration order depends on
  // static-init order across translation units, which is unspecified -- and a test
  // suite whose output ordering changes between builds is annoying to diff.
  std::sort(all.begin(), all.end(), [](const TestCase& a, const TestCase& b) {
    if (a.suite != b.suite) return a.suite < b.suite;
    return a.name < b.name;
  });

  if (list_only) {
    for (const auto& t : all) {
      std::printf("%s.%s%s\n", t.suite.c_str(), t.name.c_str(),
                  t.pending ? "  [pending]" : "");
    }
    return 0;
  }

  std::size_t passed = 0, failed = 0, pending_failed = 0, pending_passed = 0;
  std::size_t skipped = 0, filtered = 0;
  std::vector<std::string> failed_names;
  std::vector<std::string> promote_names;
  std::vector<std::string> skip_notes;

  for (const auto& t : all) {
    const std::string full = t.suite + "." + t.name;
    if (!filter.empty() && full.find(filter) == std::string::npos) {
      ++filtered;
      continue;
    }

    TestContext ctx;
    bool crashed = false;
    std::string crash_msg;

    try {
      t.fn(ctx);
    } catch (const std::exception& e) {
      crashed = true;
      crash_msg = e.what();
    } catch (...) {
      crashed = true;
      crash_msg = "unknown exception type";
    }

    // A skip wins over everything: the prerequisite was missing, so whatever the
    // assertions did before that point is not meaningful.
    if (ctx.skipped()) {
      ++skipped;
      skip_notes.push_back(full + " -- " + ctx.skip_reason());
      if (verbose) {
        std::printf("[SKIP]  %s  (%s)\n", full.c_str(), ctx.skip_reason().c_str());
      }
      continue;
    }

    const bool ok = ctx.ok() && !crashed;

    if (t.pending) {
      // Pending tests never fail the run. Their job is to tell you what is left.
      if (ok) {
        ++pending_passed;
        promote_names.push_back(full);
        std::printf("[PENDING-PASS] %s  <-- implemented! change TEST_PENDING to TEST\n",
                    full.c_str());
      } else {
        ++pending_failed;
        if (verbose) std::printf("[PENDING]      %s\n", full.c_str());
      }
      continue;
    }

    if (ok) {
      ++passed;
      if (verbose) std::printf("[PASS]  %s\n", full.c_str());
    } else {
      ++failed;
      failed_names.push_back(full);
      std::printf("[FAIL]  %s\n", full.c_str());
      if (crashed) {
        std::printf("    threw: %s\n", crash_msg.c_str());
      }
      for (const auto& f : ctx.failures()) {
        std::printf("    %s:%d\n    %s\n", f.file.c_str(), f.line, f.message.c_str());
      }
    }
  }

  std::printf("\n---------------------------------------------------------------\n");

  // A filter that selected nothing is an error, not a clean run.
  //
  // This matters because tests/CMakeLists.txt registers one ctest entry per suite via
  // --filter=<suite>. -- so if a suite is ever renamed and the CMake list is not
  // updated, that entry would run zero tests, exit 0, and report PASS forever. Silent
  // loss of coverage is the worst failure mode a test suite has, so it exits 2 here.
  //
  // Note it cannot simply check "passed == 0": a suite whose tests are all still
  // TEST_PENDING legitimately passes nothing, and tensor.* is exactly that until
  // Module 1 is written.
  const std::size_t selected =
      passed + failed + pending_failed + pending_passed + skipped;
  if (!filter.empty() && selected == 0) {
    std::printf("ERROR: --filter=%s matched none of the %zu registered tests.\n",
                filter.c_str(), all.size());
    std::printf("       Run with --list to see the real suite and test names.\n");
    std::printf("---------------------------------------------------------------\n");
    return 2;
  }

  std::printf("passed %zu   failed %zu   pending %zu   skipped %zu", passed, failed,
              pending_failed, skipped);
  if (pending_passed > 0) std::printf("   pending-but-passing %zu", pending_passed);
  if (filtered > 0) std::printf("   filtered-out %zu", filtered);
  std::printf("\n");

  if (!failed_names.empty()) {
    std::printf("\nfailed tests:\n");
    for (const auto& n : failed_names) std::printf("  %s\n", n.c_str());
  }

  if (!promote_names.empty()) {
    std::printf("\nthese pending tests now pass -- promote them to TEST:\n");
    for (const auto& n : promote_names) std::printf("  %s\n", n.c_str());
  }

  if (!skip_notes.empty()) {
    std::printf("\nskipped (missing prerequisites):\n");
    for (const auto& n : skip_notes) std::printf("  %s\n", n.c_str());
  }

  if (pending_failed > 0) {
    std::printf("\n%zu pending test(s) -- unimplemented modules. Not a failure.\n",
                pending_failed);
  }
  std::printf("---------------------------------------------------------------\n");

  return failed == 0 ? 0 : 1;
}

}  // namespace engtest

int main(int argc, char** argv) { return engtest::run_all(argc, argv); }
