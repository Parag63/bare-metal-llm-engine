#pragma once
//===----------------------------------------------------------------------===//
// engine/check.hpp -- error handling macros.
//
// WHY THIS FILE EXISTS (important if you are new to CUDA):
//
// CUDA kernel launches are ASYNCHRONOUS and they do not return an error code.
// `my_kernel<<<grid, block>>>(...)` returns immediately, before the GPU has run
// anything. That has two consequences that bite everyone once:
//
//   1. A launch configuration error (too many threads, too much shared memory)
//      is reported by the *next* CUDA call, not by the launch itself.
//   2. A fault inside the kernel (bad memory access) surfaces even later, at the
//      next synchronising call.
//
// So an error reported by `cudaMemcpy` on line 200 may really have been caused by
// a kernel launched on line 150. The fix is discipline, not cleverness:
//
//   * Wrap every CUDA API call in CUDA_CHECK(...).
//   * Call CUDA_CHECK_KERNEL() immediately after every launch in debug builds.
//
// CUDA_CHECK_KERNEL synchronises, which costs performance, so it is compiled out
// in release builds unless ENGINE_ALWAYS_SYNC_CHECK is defined.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine {

/// Exception type for all engine errors. Carries file/line so failures are traceable.
class EngineError : public std::runtime_error {
 public:
  explicit EngineError(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {
[[noreturn]] inline void throw_error(const char* file, int line, const char* expr,
                                     const std::string& msg) {
  std::ostringstream oss;
  oss << "[engine] " << file << ":" << line << "\n"
      << "  check failed: " << expr << "\n"
      << "  " << msg;
  throw EngineError(oss.str());
}
}  // namespace detail

}  // namespace engine

//===----------------------------------------------------------------------===//
// Host-side checks (always available)
//===----------------------------------------------------------------------===//

/// Unconditional runtime check. Stays enabled in release builds -- use it for
/// preconditions on public API boundaries, where the cost is irrelevant.
#define ENGINE_CHECK(cond, msg)                                              \
  do {                                                                       \
    if (!(cond)) {                                                           \
      ::engine::detail::throw_error(__FILE__, __LINE__, #cond, (msg));       \
    }                                                                        \
  } while (0)

/// Debug-only check. Compiled out with NDEBUG -- use it inside hot loops.
#ifdef NDEBUG
#define ENGINE_ASSERT(cond, msg) ((void)0)
#else
#define ENGINE_ASSERT(cond, msg) ENGINE_CHECK(cond, msg)
#endif

//===----------------------------------------------------------------------===//
// CUDA checks (only when the build has CUDA)
//===----------------------------------------------------------------------===//
#if ENGINE_HAS_CUDA

#include <cuda_runtime.h>

namespace engine {
namespace detail {
inline void check_cuda(cudaError_t err, const char* file, int line, const char* expr) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
    throw_error(file, line, expr, oss.str());
  }
}
}  // namespace detail
}  // namespace engine

/// Wrap EVERY cudaXxx() call in this.
#define CUDA_CHECK(call)                                                     \
  ::engine::detail::check_cuda((call), __FILE__, __LINE__, #call)

/// Place immediately after a kernel launch. Synchronises and reports both launch
/// errors and in-kernel faults, attributing them to the correct source line.
/// Compiled to a cheap async-only check in release builds.
#if !defined(NDEBUG) || defined(ENGINE_ALWAYS_SYNC_CHECK)
#define CUDA_CHECK_KERNEL()                                                  \
  do {                                                                       \
    CUDA_CHECK(cudaGetLastError());       /* launch-time errors */           \
    CUDA_CHECK(cudaDeviceSynchronize());  /* execution-time faults */        \
  } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

#endif  // ENGINE_HAS_CUDA
