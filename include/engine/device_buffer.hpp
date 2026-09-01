#pragma once
//===----------------------------------------------------------------------===//
// engine/device_buffer.hpp -- minimal RAII wrapper over cudaMalloc/cudaFree.
//
// SCOPE NOTE: this is deliberately throwaway scaffolding so that tests and
// benchmarks can allocate device memory today without waiting for Module 2 (the
// real memory manager, November 2026). Module 2 replaces it with a pooled
// allocator that avoids cudaMalloc in steady state. Do not build features on it.
//
// It exists because the alternative -- raw cudaMalloc/cudaFree in every test --
// leaks memory on any test failure that throws.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/config.hpp>

#if ENGINE_HAS_CUDA

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace engine {

/// Owns a device allocation of `count` elements of T. Move-only: copying a device
/// buffer implicitly would hide an expensive DtoD copy behind an `=`.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  /// Allocate and upload in one step -- the common case in tests.
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    upload(host);
  }

  ~DeviceBuffer() {
    // Destructors must not throw, so we swallow the error here rather than using
    // CUDA_CHECK. A failing cudaFree at teardown is not recoverable anyway.
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void upload(const std::vector<T>& host) {
    ENGINE_CHECK(host.size() == count_, "upload(): host/device element count mismatch");
    if (count_ > 0) {
      CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  std::vector<T> download() const {
    std::vector<T> host(count_);
    if (count_ > 0) {
      CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host;
  }

  void zero() {
    if (count_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace engine

#endif  // ENGINE_HAS_CUDA
