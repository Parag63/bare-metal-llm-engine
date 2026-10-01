#pragma once
//===----------------------------------------------------------------------===//
// tests/golden.hpp -- reader for the reference-data archive.
//
// tools/gen_reference.py runs the operations in PyTorch and writes inputs and
// expected outputs into a single binary file per test case. This reads them back.
//
// WHY A CUSTOM BINARY FORMAT rather than .npy or JSON:
//
//   * It keeps a test case in ONE file. Softmax needs an input and an expected
//     output together, and a directory of loose arrays that must be kept in sync is
//     a source of silent errors.
//   * No dependencies. A JSON parser or an .npy reader is a third-party library or
//     several hundred lines; this is about eighty.
//   * IT IS DELIBERATE PRACTICE FOR MARCH 2027. GGUF is a binary format with a magic
//     number, a version, a count, then length-prefixed names and typed tensor
//     descriptors -- structurally the same as the format below, only larger. Writing
//     this parser now means the GGUF loader is a familiar exercise rather than a new
//     skill in the middle of the hardest phase of the project.
//
// FORMAT (all little-endian, which is what x86 and NVIDIA GPUs both are):
//
//   magic       char[8]   "ENGREF01"
//   n_tensors   int32
//   repeated n_tensors times:
//     name_len  int32
//     name      char[name_len]        (not null-terminated)
//     dtype     int32                 (0 = float32; the only value used so far)
//     ndim      int32
//     dims      int64[ndim]
//     n_bytes   int64                 (redundant with dims x dtype -- and checked,
//                                      which is how corruption gets caught early)
//     data      byte[n_bytes]
//===----------------------------------------------------------------------===//

#include "test_framework.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engtest {

struct GoldenTensor {
  std::vector<std::int64_t> shape;
  std::vector<float> data;

  std::int64_t numel() const { return static_cast<std::int64_t>(data.size()); }

  /// shape[axis] with bounds checking; throws std::runtime_error on a bad axis.
  std::int64_t dim(std::size_t axis) const;
};

/// A loaded archive: named tensors from one reference file.
class GoldenFile {
 public:
  /// Loads `<golden_dir()>/<stem>.bin`. Throws std::runtime_error if the file is
  /// missing or malformed. Prefer try_load() in tests so a missing file becomes a
  /// skip rather than a failure.
  static GoldenFile load(const std::string& stem);

  /// Returns false (with `error` set) instead of throwing when the file is absent.
  static bool try_load(const std::string& stem, GoldenFile& out, std::string& error);

  /// Throws if `name` is not present -- a typo in a tensor name should be loud.
  const GoldenTensor& at(const std::string& name) const;

  bool has(const std::string& name) const;
  std::vector<std::string> names() const;
  std::size_t size() const { return tensors_.size(); }

 private:
  std::map<std::string, GoldenTensor> tensors_;
};

/// Directory holding the .bin files. Resolution order:
///   1. the ENGINE_GOLDEN_DIR environment variable, if set (handy for one-off runs)
///   2. the ENGINE_GOLDEN_DIR compile definition set by tests/CMakeLists.txt
///   3. "tests/golden" relative to the current directory
const std::string& golden_dir();

/// Human-readable instruction printed when reference data is missing. Centralised
/// so every skip message tells you the same, correct command to run.
std::string golden_missing_hint();

}  // namespace engtest

//===----------------------------------------------------------------------===//
// Loads reference data or SKIPS the test.
//
// Declares `var` as a GoldenFile in the enclosing scope. If the file is absent --
// which it is on a fresh clone, because tests/golden/ is generated data and is not
// committed -- the test is skipped with the command to run, rather than failed.
//
// The distinction matters for CI: "you have not generated the reference data yet" is
// not a broken kernel, and reporting it as one trains you to ignore red builds.
//===----------------------------------------------------------------------===//
#define LOAD_GOLDEN(var, stem)                                          \
  ::engtest::GoldenFile var;                                            \
  do {                                                                  \
    std::string engtest_err;                                            \
    if (!::engtest::GoldenFile::try_load((stem), (var), engtest_err)) { \
      SKIP_TEST(engtest_err);                                           \
    }                                                                   \
  } while (0)
