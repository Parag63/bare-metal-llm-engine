//===----------------------------------------------------------------------===//
// src/allocator.cpp -- the passthrough allocator.
//
// This is the baseline. It calls malloc/cudaMalloc directly, which is correct but
// slow. Module 2 (November 2026) adds PoolAllocator alongside it; keep this one
// forever as both a fallback and a benchmark comparison point, because "the pool is
// N times faster than raw cudaMalloc" is a number you want in the report.
//===----------------------------------------------------------------------===//

#include <engine/allocator.hpp>

#include <engine/check.hpp>
#include <engine/config.hpp>

#include <cstdlib>
#include <sstream>

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace engine {

std::string AllocStats::to_string() const {
  auto mib = [](std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << "in_use=" << mib(bytes_in_use) << " MiB"
      << ", reserved=" << mib(bytes_reserved) << " MiB"
      << ", peak=" << mib(peak_bytes_in_use) << " MiB"
      << ", allocs=" << num_allocs
      << ", driver_allocs=" << num_driver_allocs;
  return oss.str();
}

namespace {

/// Direct passthrough. Tracks stats so it is comparable with the future pool.
class PassthroughAllocator final : public Allocator {
 public:
  explicit PassthroughAllocator(Device device) : device_(device) {}

  void* allocate(std::size_t bytes) override {
    if (bytes == 0) return nullptr;

    void* p = nullptr;
    if (device_ == Device::CPU) {
      // 256-byte alignment matches what cudaMalloc guarantees, so host and device
      // buffers behave the same way with respect to vectorised loads (float4 needs
      // 16-byte alignment; 256 covers every case we will meet).
#if defined(_MSC_VER)
      p = _aligned_malloc(round_up(bytes, 256), 256);
#else
      p = std::aligned_alloc(256, round_up(bytes, 256));
#endif
      ENGINE_CHECK(p != nullptr, "host allocation failed");
    } else {
#if ENGINE_HAS_CUDA
      CUDA_CHECK(cudaMalloc(&p, bytes));
#else
      ENGINE_CHECK(false, "CUDA allocator requested in a CPU-only build");
#endif
    }

    stats_.bytes_in_use += bytes;
    stats_.bytes_reserved += bytes;
    stats_.peak_bytes_in_use =
        (stats_.bytes_in_use > stats_.peak_bytes_in_use) ? stats_.bytes_in_use
                                                         : stats_.peak_bytes_in_use;
    ++stats_.num_allocs;
    ++stats_.num_driver_allocs;  // every request hits the driver -- the whole problem
    return p;
  }

  void deallocate(void* ptr) override {
    if (ptr == nullptr) return;
    if (device_ == Device::CPU) {
#if defined(_MSC_VER)
      _aligned_free(ptr);
#else
      std::free(ptr);
#endif
    } else {
#if ENGINE_HAS_CUDA
      cudaFree(ptr);
#endif
    }
    // NOTE: bytes_in_use is not decremented, because a bare pointer does not tell
    // us its size. The real fix is a size map, which PoolAllocator will need
    // anyway -- so this is left honest-but-incomplete rather than wrong-and-hidden.
  }

  Device device() const override { return device_; }
  AllocStats stats() const override { return stats_; }

 private:
  static std::size_t round_up(std::size_t v, std::size_t a) {
    return ((v + a - 1) / a) * a;
  }

  Device device_;
  AllocStats stats_{};
};

}  // namespace

Allocator& default_allocator(Device device) {
  // Function-local statics: thread-safe initialisation since C++11, and they are
  // constructed on first use rather than at program start, which matters because
  // touching the CUDA runtime during static init is a known source of trouble.
  static PassthroughAllocator cpu_alloc(Device::CPU);
  static PassthroughAllocator cuda_alloc(Device::CUDA);
  return (device == Device::CPU) ? static_cast<Allocator&>(cpu_alloc)
                                 : static_cast<Allocator&>(cuda_alloc);
}

}  // namespace engine
