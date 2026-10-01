//===----------------------------------------------------------------------===//
// tests/test_allocator.cpp -- Validate PoolAllocator and memory accounting.
//===----------------------------------------------------------------------===//

#include "test_framework.hpp"

#include <engine/check.hpp>
#include <engine/cuda_device.hpp>
#include <engine/pool_allocator.hpp>

#include <cstdint>
#include <vector>

namespace {

TEST(allocator, pool_allocates_aligned_blocks) {
  engine::PoolAllocator pool(engine::Device::CPU, 64 * 1024);

  void* p1 = pool.allocate(100);
  ASSERT_TRUE(p1 != nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p1) % 256, 0u);

  void* p2 = pool.allocate(5000);
  ASSERT_TRUE(p2 != nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p2) % 256, 0u);

  pool.deallocate(p1);
  pool.deallocate(p2);
}

TEST(allocator, pool_recycles_freed_memory_blocks) {
  engine::PoolAllocator pool(engine::Device::CPU, 64 * 1024);

  void* p1 = pool.allocate(1024);
  ASSERT_TRUE(p1 != nullptr);
  const auto stats1 = pool.stats();
  EXPECT_EQ(stats1.num_driver_allocs, 1u);
  EXPECT_EQ(stats1.num_allocs, 1u);

  pool.deallocate(p1);
  const auto stats2 = pool.stats();
  EXPECT_EQ(stats2.bytes_in_use, 0u);

  // Second allocation of same size bucket should recycle p1 without a new driver alloc
  void* p2 = pool.allocate(1024);
  EXPECT_EQ(p1, p2);
  const auto stats3 = pool.stats();
  EXPECT_EQ(stats3.num_driver_allocs, 1u);
  EXPECT_EQ(stats3.num_allocs, 2u);

  pool.deallocate(p2);
}

TEST(allocator, tracks_exact_bytes_in_use_and_peak) {
  engine::PoolAllocator pool(engine::Device::CPU, 64 * 1024);

  void* a = pool.allocate(1024);  // rounded to 1024
  void* b = pool.allocate(2048);  // rounded to 2048
  auto s1 = pool.stats();
  EXPECT_EQ(s1.bytes_in_use, 3072u);
  EXPECT_EQ(s1.peak_bytes_in_use, 3072u);

  pool.deallocate(a);
  auto s2 = pool.stats();
  EXPECT_EQ(s2.bytes_in_use, 2048u);
  EXPECT_EQ(s2.peak_bytes_in_use, 3072u);  // peak stays at 3072

  void* c = pool.allocate(4096);
  auto s3 = pool.stats();
  EXPECT_EQ(s3.bytes_in_use, 6144u);
  EXPECT_EQ(s3.peak_bytes_in_use, 6144u);

  pool.deallocate(b);
  pool.deallocate(c);
  auto s4 = pool.stats();
  EXPECT_EQ(s4.bytes_in_use, 0u);
  EXPECT_EQ(s4.peak_bytes_in_use, 6144u);
}

TEST(allocator, acceptance_test_100k_cycles_driver_allocs_under_20) {
  engine::PoolAllocator pool(engine::Device::CPU, 64 * 1024);

  // Mentor's required acceptance test: 100,000 alternating allocate/deallocate cycles
  // must produce num_driver_allocs < 20.
  for (int i = 0; i < 100000; ++i) {
    void* p = pool.allocate(512);
    pool.deallocate(p);
  }

  const auto stats = pool.stats();
  EXPECT_EQ(stats.num_allocs, 100000u);
  EXPECT_TRUE(stats.num_driver_allocs < 20u);
  EXPECT_EQ(stats.num_driver_allocs, 1u);
  EXPECT_EQ(stats.bytes_in_use, 0u);
}

#if ENGINE_HAS_CUDA
TEST(allocator, cuda_pool_allocates_and_recycles_device_memory) {
  if (engine::cuda_device_count() == 0) return;

  engine::PoolAllocator pool(engine::Device::CUDA, 1024 * 1024);

  void* d1 = pool.allocate(4096);
  ASSERT_TRUE(d1 != nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(d1) % 256, 0u);

  pool.deallocate(d1);
  void* d2 = pool.allocate(4096);
  EXPECT_EQ(d1, d2);

  const auto stats = pool.stats();
  EXPECT_EQ(stats.num_driver_allocs, 1u);
  EXPECT_EQ(stats.num_allocs, 2u);

  pool.deallocate(d2);
}
#endif

}  // namespace
