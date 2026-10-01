#pragma once
//===----------------------------------------------------------------------===//
// engine/cuda_stream.hpp -- RAII wrappers for cudaStream_t and cudaEvent_t.
//
// Provides zero-overhead abstractions for asynchronous GPU scheduling,
// multi-stream concurrency, and precise CUDA event timing.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>
#include <engine/check.hpp>

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#include <utility>

namespace engine {

/// RAII wrapper managing the lifecycle of a cudaStream_t.
class CudaStream {
 public:
  /// Creates a non-blocking stream by default (cudaStreamNonBlocking).
  CudaStream();

  /// Explicit flag constructor (e.g. cudaStreamDefault).
  explicit CudaStream(unsigned int flags);

  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_) {
        cudaStreamDestroy(stream_);
      }
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; }

  void synchronize() const {
    if (stream_) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

 private:
  cudaStream_t stream_ = nullptr;
};

/// RAII wrapper managing the lifecycle of a cudaEvent_t.
class CudaEvent {
 public:
  /// Creates a timing event by default (cudaEventDefault).
  CudaEvent();

  /// Explicit flag constructor (e.g. cudaEventDisableTiming).
  explicit CudaEvent(unsigned int flags);

  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_) {
        cudaEventDestroy(event_);
      }
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  cudaEvent_t get() const { return event_; }
  operator cudaEvent_t() const { return event_; }

  void record(cudaStream_t stream = 0) {
    ENGINE_CHECK(event_ != nullptr, "CudaEvent::record called on null event");
    CUDA_CHECK(cudaEventRecord(event_, stream));
  }

  void synchronize() const {
    if (event_) {
      CUDA_CHECK(cudaEventSynchronize(event_));
    }
  }

  static float elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    ENGINE_CHECK(start.get() != nullptr && end.get() != nullptr,
                 "CudaEvent::elapsed_ms called on uninitialized event");
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), end.get()));
    return ms;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace engine

#endif  // ENGINE_HAS_CUDA
