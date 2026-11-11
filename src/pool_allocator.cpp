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

void* PoolAllocator::record_allocation(void* ptr, std::size_t bucket_size) {
  stats_.bytes_in_use += bucket_size;
  stats_.peak_bytes_in_use = std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
  stats_.num_allocs++;
  allocated_sizes_[ptr] = bucket_size;
  return ptr;
}

#if ENGINE_HAS_CUDA
cudaEvent_t PoolAllocator::acquire_event() {
  if (!event_pool_.empty()) {
    cudaEvent_t ev = event_pool_.back();
    event_pool_.pop_back();
    return ev;
  }
  cudaEvent_t ev = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  return ev;
}

void PoolAllocator::release_event(cudaEvent_t event) {
  if (event != nullptr) {
    event_pool_.push_back(event);
  }
}
#endif

void* PoolAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;

  const std::size_t bucket_size = round_up_power_of_two(bytes);

  std::lock_guard<std::mutex> lock(mutex_);

  // 1. Check size-class free queue; skip CUDA blocks whose stream event is still pending.
  auto it = free_lists_.find(bucket_size);
  if (it != free_lists_.end()) {
    auto& queue = it->second;
    for (auto blk_it = queue.begin(); blk_it != queue.end(); ++blk_it) {
      bool ready = true;
#if ENGINE_HAS_CUDA
      if (blk_it->ready_event != nullptr) {
        const cudaError_t q = cudaEventQuery(blk_it->ready_event);
        if (q == cudaSuccess) {
          release_event(blk_it->ready_event);
          blk_it->ready_event = nullptr;
        } else if (q == cudaErrorNotReady) {
          ready = false;
        } else {
          release_event(blk_it->ready_event);
          blk_it->ready_event = nullptr;
          CUDA_CHECK(q);
        }
      }
#endif
      if (ready) {
        void* p = blk_it->ptr;
        queue.erase(blk_it);
        return record_allocation(p, bucket_size);
      }
    }
  }

  // 2. Carve from existing slabs if space permits (scan most recent first so one-off
  // large slabs do not strand remaining space in earlier slabs).
  for (auto slab_it = slabs_.rbegin(); slab_it != slabs_.rend(); ++slab_it) {
    const std::size_t remaining = slab_it->size - slab_it->offset;
    if (remaining >= bucket_size) {
      char* base = static_cast<char*>(slab_it->ptr);
      void* p = base + slab_it->offset;
      slab_it->offset += bucket_size;
      return record_allocation(p, bucket_size);
    }
  }

  // 3. Allocate a new slab from the driver. Cap automatic slab doubling so a single
  // huge allocation (e.g., 2 GiB naive score matrix) does not force the next tiny
  // allocation to reserve a 4 GiB slab.
  static constexpr std::size_t kMaxAutoSlabSize = std::size_t{64} * 1024 * 1024;
  const std::size_t slab_size = std::max(next_slab_size_, bucket_size);
  next_slab_size_ = std::min(kMaxAutoSlabSize, next_slab_size_ * 2);

  void* slab_ptr = driver_alloc(slab_size);
  stats_.bytes_reserved += slab_size;
  stats_.num_driver_allocs++;

  slabs_.push_back({slab_ptr, slab_size, bucket_size});
  return record_allocation(slab_ptr, bucket_size);
}

void PoolAllocator::deallocate(void* ptr) {
  if (!ptr) return;

#if ENGINE_HAS_CUDA
  if (device_ == Device::CUDA) {
    deallocate(ptr, static_cast<cudaStream_t>(nullptr));
    return;
  }
#endif

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocated_sizes_.find(ptr);
  ENGINE_CHECK(it != allocated_sizes_.end(),
               "PoolAllocator: pointer was not allocated by this pool");

  const std::size_t bucket_size = it->second;
  allocated_sizes_.erase(it);

  stats_.bytes_in_use -= bucket_size;
  free_lists_[bucket_size].push_back(FreeBlock{ptr});
}

#if ENGINE_HAS_CUDA
void PoolAllocator::deallocate(void* ptr, cudaStream_t stream) {
  if (!ptr) return;
  if (device_ != Device::CUDA) {
    deallocate(ptr);
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocated_sizes_.find(ptr);
  ENGINE_CHECK(it != allocated_sizes_.end(),
               "PoolAllocator: pointer was not allocated by this pool");

  const std::size_t bucket_size = it->second;
  cudaEvent_t ev = acquire_event();
  const cudaError_t err = cudaEventRecord(ev, stream);
  if (err != cudaSuccess) {
    release_event(ev);
    CUDA_CHECK(err);
  }

  allocated_sizes_.erase(it);
  stats_.bytes_in_use -= bucket_size;
  free_lists_[bucket_size].push_back(FreeBlock{ptr, ev});
}
#endif

AllocStats PoolAllocator::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void PoolAllocator::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

#if ENGINE_HAS_CUDA
  for (auto& kv : free_lists_) {
    for (auto& blk : kv.second) {
      if (blk.ready_event != nullptr) {
        cudaEventDestroy(blk.ready_event);
      }
    }
  }
  for (cudaEvent_t ev : event_pool_) {
    cudaEventDestroy(ev);
  }
  event_pool_.clear();
#endif

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
