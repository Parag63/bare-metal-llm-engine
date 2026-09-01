#include "golden.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace engtest {

namespace {

constexpr char kMagic[8] = {'E', 'N', 'G', 'R', 'E', 'F', '0', '1'};
constexpr std::int32_t kDTypeF32 = 0;

[[noreturn]] void fail(const std::string& path, const std::string& why) {
  throw std::runtime_error("golden file '" + path + "': " + why);
}

/// Reads a fixed-size little-endian value.
///
/// Reading straight into the object's bytes is correct on x86 and ARM, which are
/// both little-endian, and this project has no big-endian target. The size check
/// still matters: a truncated file would otherwise leave the value partly
/// uninitialised and produce a confusing downstream error instead of a clear one.
template <typename T>
T read_pod(std::istream& is, const std::string& path, const char* what) {
  T value{};
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!is) fail(path, std::string("unexpected end of file while reading ") + what);
  return value;
}

}  // namespace

std::int64_t GoldenTensor::dim(std::size_t axis) const {
  if (axis >= shape.size()) {
    throw std::runtime_error("GoldenTensor::dim(): axis " + std::to_string(axis) +
                             " out of range for a " + std::to_string(shape.size()) +
                             "-d tensor");
  }
  return shape[axis];
}

const std::string& golden_dir() {
  static const std::string dir = [] {
    if (const char* env = std::getenv("ENGINE_GOLDEN_DIR")) {
      return std::string(env);
    }
#ifdef ENGINE_GOLDEN_DIR_DEFAULT
    return std::string(ENGINE_GOLDEN_DIR_DEFAULT);
#else
    return std::string("tests/golden");
#endif
  }();
  return dir;
}

std::string golden_missing_hint() {
  return "reference data not generated yet -- run:  python3 tools/gen_reference.py";
}

bool GoldenFile::try_load(const std::string& stem, GoldenFile& out, std::string& error) {
  try {
    out = load(stem);
    return true;
  } catch (const std::exception& e) {
    error = e.what();
    return false;
  }
}

GoldenFile GoldenFile::load(const std::string& stem) {
  const std::string path = golden_dir() + "/" + stem + ".bin";

  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("cannot open '" + path + "' -- " + golden_missing_hint());
  }

  char magic[8] = {};
  is.read(magic, sizeof(magic));
  if (!is || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    fail(path, "bad magic -- not an ENGREF01 file (regenerate it)");
  }

  const std::int32_t n_tensors = read_pod<std::int32_t>(is, path, "n_tensors");
  if (n_tensors < 0 || n_tensors > 4096) {
    fail(path, "implausible tensor count " + std::to_string(n_tensors));
  }

  GoldenFile result;

  for (std::int32_t t = 0; t < n_tensors; ++t) {
    const std::int32_t name_len = read_pod<std::int32_t>(is, path, "name_len");
    if (name_len <= 0 || name_len > 1024) {
      fail(path, "implausible name length " + std::to_string(name_len));
    }

    std::string name(static_cast<std::size_t>(name_len), '\0');
    is.read(name.data(), name_len);
    if (!is) fail(path, "truncated tensor name");

    const std::int32_t dtype = read_pod<std::int32_t>(is, path, "dtype");
    if (dtype != kDTypeF32) {
      fail(path, "tensor '" + name + "' has dtype " + std::to_string(dtype) +
                     "; only float32 (0) is supported so far");
    }

    const std::int32_t ndim = read_pod<std::int32_t>(is, path, "ndim");
    if (ndim < 0 || ndim > 8) {
      fail(path, "implausible ndim " + std::to_string(ndim) + " for '" + name + "'");
    }

    GoldenTensor tensor;
    tensor.shape.resize(static_cast<std::size_t>(ndim));
    std::int64_t expected_elems = 1;
    for (std::int32_t d = 0; d < ndim; ++d) {
      const std::int64_t v = read_pod<std::int64_t>(is, path, "dim");
      if (v < 0) fail(path, "negative dimension in '" + name + "'");
      tensor.shape[static_cast<std::size_t>(d)] = v;
      expected_elems *= v;
    }

    const std::int64_t n_bytes = read_pod<std::int64_t>(is, path, "n_bytes");

    // The redundancy check. dims and n_bytes are written independently by the
    // generator, so a mismatch means the file is corrupt or the writer and reader
    // have drifted apart -- exactly the bug this catches cheaply.
    const std::int64_t implied = expected_elems * static_cast<std::int64_t>(sizeof(float));
    if (n_bytes != implied) {
      fail(path, "tensor '" + name + "': header says " + std::to_string(n_bytes) +
                     " bytes but its shape implies " + std::to_string(implied));
    }

    tensor.data.resize(static_cast<std::size_t>(expected_elems));
    if (expected_elems > 0) {
      is.read(reinterpret_cast<char*>(tensor.data.data()), n_bytes);
      if (!is) fail(path, "truncated data for '" + name + "'");
    }

    if (!result.tensors_.emplace(name, std::move(tensor)).second) {
      fail(path, "duplicate tensor name '" + name + "'");
    }
  }

  // Trailing bytes mean the writer emitted something the reader does not understand,
  // i.e. the two have diverged. Better to say so than to ignore it.
  is.peek();
  if (!is.eof()) {
    fail(path, "unexpected trailing bytes -- writer and reader formats disagree");
  }

  return result;
}

const GoldenTensor& GoldenFile::at(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    std::ostringstream oss;
    oss << "golden tensor '" << name << "' not found. Available:";
    for (const auto& kv : tensors_) oss << " " << kv.first;
    throw std::runtime_error(oss.str());
  }
  return it->second;
}

bool GoldenFile::has(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> GoldenFile::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& kv : tensors_) out.push_back(kv.first);
  return out;
}

}  // namespace engtest
