#pragma once
//===----------------------------------------------------------------------===//
// engine/pool_allocator.hpp -- High-performance slab & bucket pooled allocator.
//
// Eliminates runtime cudaMalloc / cudaFree driver synchronization overhead
// during autoregressive decode loops by caching memory blocks in power-of-two
// size buckets carved from large contiguous slabs.
//===----------------------------------------------------------------------===//

#include <engine/allocator.hpp>

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

  Device device_;
  std::size_t initial_slab_size_;
  std::size_t next_slab_size_;

  mutable std::mutex mutex_;
  AllocStats stats_;

  // Free lists per power-of-two size bucket
  std::map<std::size_t, std::vector<void*>> free_lists_;
  // Active allocations mapping ptr -> bucket size
  std::unordered_map<void*, std::size_t> allocated_sizes_;

  // Slabs reserved from the driver
  struct Slab {
    void* ptr = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;
  };
  std::vector<Slab> slabs_;
};

/// Returns the singleton PoolAllocator instance for the specified device.
PoolAllocator& pool_allocator(Device device);

}  // namespace engine
