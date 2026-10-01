#pragma once
//===----------------------------------------------------------------------===//
// engine/allocator.hpp -- Module 2, the memory manager (November 2026).
//
// SCHEDULED FOR NOVEMBER 2026. Declared now only so Storage in tensor.hpp can be
// written against a stable interface and later swapped from raw cudaMalloc to a
// pool without touching Tensor at all.
//
// WHY A POOL IS NEEDED (the justification to put in your report):
//
// cudaMalloc is not a fast allocator. It synchronises the device and can take tens
// of microseconds. During autoregressive generation you allocate and free
// activation buffers on EVERY token of EVERY layer -- for a 32-layer model
// producing 512 tokens that is tens of thousands of allocations. At even 20 us
// each, allocation alone would dominate your inference time.
//
// The fix is the standard one: grab a few large slabs from cudaMalloc once at
// startup, then satisfy all subsequent requests by bumping a pointer within a slab
// and recycling freed blocks. Steady-state allocation becomes a handful of
// instructions and zero driver calls.
//
// This is also where PagedAttention-style KV-cache management would eventually
// live, if you extend the project past the current scope.
//===----------------------------------------------------------------------===//

#include <engine/tensor.hpp>

#include <cstddef>
#include <string>

namespace engine {

/// Allocation statistics, for the memory numbers in the benchmark report.
struct AllocStats {
  std::size_t bytes_in_use = 0;       ///< currently handed out to callers
  std::size_t bytes_reserved = 0;     ///< total obtained from the driver
  std::size_t peak_bytes_in_use = 0;  ///< high-water mark; this is the number to report
  std::size_t num_allocs = 0;         ///< allocate() calls served
  std::size_t num_driver_allocs = 0;  ///< times we actually called cudaMalloc.
                                      ///< The pool works iff this stays tiny.
  std::string to_string() const;
};

/// Interface every allocator implements, so Storage can be device-agnostic.
class Allocator {
 public:
  virtual ~Allocator() = default;

  /// Returns a pointer to at least `bytes`, aligned to at least 256 bytes.
  /// (256 because that is the alignment cudaMalloc guarantees and what coalesced
  /// vectorised loads such as float4 require.)
  virtual void* allocate(std::size_t bytes) = 0;

  virtual void deallocate(void* ptr) = 0;

  virtual Device device() const = 0;

  virtual AllocStats stats() const = 0;
};

/// Straight passthrough to malloc/cudaMalloc. Correct but slow; the baseline the
/// pooled allocator must beat, and the fallback if the pool has a bug.
Allocator& default_allocator(Device device);

/// TODO(parag, Nov 2026): PoolAllocator.
///   Suggested design, simplest thing that works:
///     - Round every request up to a power of two (min 256 B).
///     - Keep a free list per size class.
///     - On miss, carve from the current slab; if the slab is exhausted, request a
///       new one from the driver at 2x the previous size.
///     - deallocate() pushes onto the size-class free list; never returns to the
///       driver during a run.
///   Acceptance test: 100k alternating allocate/deallocate cycles must produce
///   num_driver_allocs < 20. That single assertion is the whole point of the module.

}  // namespace engine
