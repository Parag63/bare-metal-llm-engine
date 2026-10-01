//===----------------------------------------------------------------------===//
// src/pool_allocator.cpp -- implementation of PoolAllocator.
//===----------------------------------------------------------------------===//

#include <engine/pool_allocator.hpp>

#include <engine/check.hpp>
#include <engine/config.hpp>

#include <algorithm>
#include <cstdlib>

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#endif

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace engine {

namespace {

std::size_t round_up_aligned(std::size_t n, std::size_t align) {
  return (n + align - 1) & ~(align - 1);
}

}  // namespace

PoolAllocator::PoolAllocator(Device device, std::size_t initial_slab_size)
    : device_(device),
      initial_slab_size_(std::max(initial_slab_size, kMinAllocSize)),
      next_slab_size_(initial_slab_size_) {}

PoolAllocator::~PoolAllocator() { reset(); }

std::size_t PoolAllocator::round_up_power_of_two(std::size_t bytes) {
  if (bytes <= kMinAllocSize) return kMinAllocSize;
  std::size_t v = bytes - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

void* PoolAllocator::driver_alloc(std::size_t bytes) {
  void* p = nullptr;
  if (device_ == Device::CPU) {
#if defined(_MSC_VER)
    p = _aligned_malloc(round_up_aligned(bytes, kAlignment), kAlignment);
#else
    p = std::aligned_alloc(kAlignment, round_up_aligned(bytes, kAlignment));
#endif
    ENGINE_CHECK(p != nullptr, "PoolAllocator: host aligned_alloc failed for " +
                                   std::to_string(bytes) + " bytes");
  } else if (device_ == Device::CUDA) {
#if ENGINE_HAS_CUDA
    CUDA_CHECK(cudaMalloc(&p, bytes));
#else
    ENGINE_CHECK(false, "PoolAllocator: CUDA requested but engine built without CUDA");
#endif
  }
  return p;
}

void PoolAllocator::driver_free(void* ptr) {
  if (!ptr) return;
  if (device_ == Device::CPU) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
  } else if (device_ == Device::CUDA) {
#if ENGINE_HAS_CUDA
    cudaFree(ptr);
#endif
  }
}

void* PoolAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;

  const std::size_t bucket_size = round_up_power_of_two(bytes);

  std::lock_guard<std::mutex> lock(mutex_);

  // 1. Check size-class free list
  auto it = free_lists_.find(bucket_size);
  if (it != free_lists_.end() && !it->second.empty()) {
    void* p = it->second.back();
    it->second.pop_back();

    stats_.bytes_in_use += bucket_size;
    stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    stats_.num_allocs++;
    allocated_sizes_[p] = bucket_size;
    return p;
  }

  // 2. Carve from existing slab if space permits
  if (!slabs_.empty()) {
    Slab& current = slabs_.back();
    const std::size_t remaining = current.size - current.offset;
    if (remaining >= bucket_size) {
      char* base = static_cast<char*>(current.ptr);
      void* p = base + current.offset;
      current.offset += bucket_size;

      stats_.bytes_in_use += bucket_size;
      stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.num_allocs++;
      allocated_sizes_[p] = bucket_size;
      return p;
    }
  }

  // 3. Allocate a new slab from the driver
  const std::size_t slab_size = std::max(next_slab_size_, bucket_size);
  next_slab_size_ = std::max(next_slab_size_ * 2, slab_size * 2);

  void* slab_ptr = driver_alloc(slab_size);
  stats_.bytes_reserved += slab_size;
  stats_.num_driver_allocs++;

  slabs_.push_back({slab_ptr, slab_size, bucket_size});

  stats_.bytes_in_use += bucket_size;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
  stats_.num_allocs++;
  allocated_sizes_[slab_ptr] = bucket_size;

  return slab_ptr;
}

void PoolAllocator::deallocate(void* ptr) {
  if (!ptr) return;

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocated_sizes_.find(ptr);
  ENGINE_CHECK(it != allocated_sizes_.end(),
               "PoolAllocator: pointer was not allocated by this pool");

  const std::size_t bucket_size = it->second;
  allocated_sizes_.erase(it);

  stats_.bytes_in_use -= bucket_size;
  free_lists_[bucket_size].push_back(ptr);
}

AllocStats PoolAllocator::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void PoolAllocator::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& slab : slabs_) {
    driver_free(slab.ptr);
  }
  slabs_.clear();
  free_lists_.clear();
  allocated_sizes_.clear();

  stats_ = AllocStats{};
  next_slab_size_ = initial_slab_size_;
}

PoolAllocator& pool_allocator(Device device) {
  if (device == Device::CPU) {
    static PoolAllocator cpu_pool(Device::CPU);
    return cpu_pool;
  }
#if ENGINE_HAS_CUDA
  if (device == Device::CUDA) {
    static PoolAllocator cuda_pool(Device::CUDA);
    return cuda_pool;
  }
#endif
  ENGINE_CHECK(false, "pool_allocator: unsupported device");
}

}  // namespace engine
