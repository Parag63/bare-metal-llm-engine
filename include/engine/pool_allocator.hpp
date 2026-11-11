#pragma once
//===----------------------------------------------------------------------===//
// engine/pool_allocator.hpp -- High-performance slab & bucket pooled allocator.
//
// Eliminates runtime cudaMalloc / cudaFree driver synchronization overhead
// during autoregressive decode loops by caching memory blocks in power-of-two
// size buckets carved from large contiguous slabs.
//===----------------------------------------------------------------------===//

#include <engine/allocator.hpp>
#include <engine/config.hpp>

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace engine {

/// High-performance pooled allocator with exact memory accounting.
class PoolAllocator final : public Allocator {
 public:
  /// Constructs a pool allocator for the given device with an initial slab size.
  explicit PoolAllocator(Device device, std::size_t initial_slab_size = 4 * 1024 * 1024);
  ~PoolAllocator() override;

  PoolAllocator(const PoolAllocator&) = delete;
  PoolAllocator& operator=(const PoolAllocator&) = delete;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;
#if ENGINE_HAS_CUDA
  /// Returns `ptr` to the pool once all work queued on `stream` finishes.
  /// Records an event on `stream` and defers reuse until cudaEventQuery succeeds.
  void deallocate(void* ptr, cudaStream_t stream);
#endif
  Device device() const override { return device_; }
  AllocStats stats() const override;

  /// Releases all slabs back to the OS / GPU driver and resets statistics.
  void reset();

 private:
  static constexpr std::size_t kMinAllocSize = 256;
  static constexpr std::size_t kAlignment = 256;

  static std::size_t round_up_power_of_two(std::size_t bytes);

  void* driver_alloc(std::size_t bytes);
  void driver_free(void* ptr);
  void* record_allocation(void* ptr, std::size_t bucket_size);
#if ENGINE_HAS_CUDA
  cudaEvent_t acquire_event();
  void release_event(cudaEvent_t event);
#endif

  Device device_;
  std::size_t initial_slab_size_;
  std::size_t next_slab_size_;

  mutable std::mutex mutex_;
  AllocStats stats_;

  struct FreeBlock {
    void* ptr = nullptr;
#if ENGINE_HAS_CUDA
    cudaEvent_t ready_event = nullptr;
#endif
  };

  // Free queues per power-of-two bucket; CUDA blocks carry their completion event.
  std::map<std::size_t, std::deque<FreeBlock>> free_lists_;
  // Active allocations mapping ptr -> bucket size
  std::unordered_map<void*, std::size_t> allocated_sizes_;
#if ENGINE_HAS_CUDA
  std::vector<cudaEvent_t> event_pool_;
#endif

  // Slabs reserved from the driver
  struct Slab {
    void* ptr = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;
  };
  std::vector<Slab> slabs_;
};

/// RAII guard for scratch memory allocated from a PoolAllocator.
/// On CUDA pools, binds deallocation to the launch stream so the block is not
/// reused by another stream until prior kernels finish.
class ScopedPoolBuffer {
 public:
  ScopedPoolBuffer(PoolAllocator& pool, std::size_t bytes)
      : pool_(pool), ptr_(pool.allocate(bytes)) {}

#if ENGINE_HAS_CUDA
  ScopedPoolBuffer(PoolAllocator& pool, std::size_t bytes, cudaStream_t stream)
      : pool_(pool), ptr_(pool.allocate(bytes)), stream_(stream) {}
#endif

  ~ScopedPoolBuffer() {
    try {
      release();
    } catch (...) {
    }
  }

  ScopedPoolBuffer(const ScopedPoolBuffer&) = delete;
  ScopedPoolBuffer& operator=(const ScopedPoolBuffer&) = delete;
  ScopedPoolBuffer(ScopedPoolBuffer&&) = delete;
  ScopedPoolBuffer& operator=(ScopedPoolBuffer&&) = delete;

  void* get() const { return ptr_; }

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

  void release() {
    if (ptr_ == nullptr) return;
    void* p = ptr_;
    ptr_ = nullptr;
#if ENGINE_HAS_CUDA
    if (pool_.device() == Device::CUDA) {
      pool_.deallocate(p, stream_);
      return;
    }
#endif
    pool_.deallocate(p);
  }

 private:
  PoolAllocator& pool_;
  void* ptr_ = nullptr;
#if ENGINE_HAS_CUDA
  cudaStream_t stream_ = nullptr;
#endif
};

/// Returns the singleton PoolAllocator instance for the specified device.
PoolAllocator& pool_allocator(Device device);

}  // namespace engine
