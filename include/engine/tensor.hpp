#pragma once
//===----------------------------------------------------------------------===//
// engine/tensor.hpp -- Module 1, the tensor abstraction (Objective 1).
//
// IMPLEMENTED. All 10 steps complete in src/tensor.cpp (559 lines). 24 tensor
// tests + 4 storage tests passing. See ADR 0002 for the Storage/Tensor split.
//
//===----------------------------------------------------------------------===//
// THE ONE DESIGN DECISION THAT MATTERS HERE
//
// A Tensor is split into two objects:
//
//   Storage -- reference-counted ownership of a flat byte buffer on one device.
//   Tensor  -- a VIEW onto a Storage: shape, strides, byte offset, dtype.
//
// Multiple Tensors can share one Storage. This is exactly what PyTorch does, and
// you need it, because three things later in this project are fundamentally views:
//
//   * KV-cache slices (Feb 2027): each generation step reads cache[:, :pos, :].
//     Copying instead of viewing would defeat the entire point of having a cache.
//   * Weight tensors from a memory-mapped GGUF file (Mar 2027): ~50 tensors must
//     alias one mmap region, or you double peak memory on a 4 GB model.
//   * Reshapes in attention: [B, S, H*D] -> [B, S, H, D] must be free.
//
// Retrofitting views onto a tensor that owns its buffer outright is a painful
// refactor. Paying for the indirection now is the cheaper choice. See ADR 0002.
//===----------------------------------------------------------------------===//
// STRIDES, IF THEY ARE NEW TO YOU
//
// stride[i] is how many ELEMENTS you skip to advance one step along axis i.
// For a contiguous row-major [2,3,4] tensor: strides = [12, 4, 1].
// Address of (i,j,k) = offset + i*12 + j*4 + k*1.
//
// Strides make transpose and slicing O(1) metadata edits instead of data copies:
// transposing swaps two strides and touches no memory. The cost is that data is
// then non-contiguous, which most CUDA kernels cannot handle -- hence
// is_contiguous() and contiguous(), and hence why kernels assert contiguity.
//===----------------------------------------------------------------------===//

#include <engine/dtype.hpp>
#include <engine/half.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine {

enum class Device : std::uint8_t {
  CPU = 0,
  CUDA = 1,
};

const char* device_name(Device d);

/// See the DType overload in dtype.hpp: prints "cpu"/"cuda" instead of a number.
inline std::ostream& operator<<(std::ostream& os, Device d) {
  return os << device_name(d);
}

//===----------------------------------------------------------------------===//
// Storage: reference-counted raw buffer on a single device.
//===----------------------------------------------------------------------===//

/// Tag type for constructing a non-owning Storage (see Tensor::from_blob).
struct BorrowTag {};

class Storage {
 public:
  /// Allocates `bytes` on `device`. Throws EngineError on allocation failure, and
  /// throws if device == CUDA on a CPU-only build.
  Storage(std::size_t bytes, Device device);

  /// Non-owning constructor: wraps an externally-owned pointer without allocating.
  /// The destructor will NOT free this pointer. The caller guarantees the pointer
  /// outlives every Storage/Tensor that references it.
  /// This is what lets ~50 weight tensors alias one mmap'd GGUF region without
  /// copying (March 2027).
  Storage(void* data, std::size_t bytes, Device device, BorrowTag);

  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;  // shared via shared_ptr; never moved
  Storage& operator=(Storage&&) = delete;

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }
  Device device() const { return device_; }
  bool owns() const { return owns_; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  Device device_ = Device::CPU;
  bool owns_ = true;
};

//===----------------------------------------------------------------------===//
// Tensor: an N-dimensional view onto a Storage.
//===----------------------------------------------------------------------===//
class Tensor {
 public:
  /// Empty tensor: no storage, numel() == 0.
  Tensor() = default;

  /// Allocates a new contiguous tensor. Contents are UNINITIALISED (deliberately --
  /// zeroing multi-gigabyte weight buffers you are about to overwrite is pure waste).
  Tensor(std::vector<std::int64_t> shape, DType dtype, Device device);

  /// Zero-filled factory.
  static Tensor zeros(std::vector<std::int64_t> shape, DType dtype, Device device);

  /// Wraps EXISTING memory without taking ownership. The caller guarantees the
  /// pointer outlives the tensor. Needed for mmap'd GGUF weights.
  static Tensor from_blob(void* data, std::vector<std::int64_t> shape, DType dtype,
                          Device device);

  // --- shape / layout -------------------------------------------------------
  const std::vector<std::int64_t>& shape() const { return shape_; }
  const std::vector<std::int64_t>& strides() const { return strides_; }
  std::int64_t dim() const { return static_cast<std::int64_t>(shape_.size()); }
  std::int64_t size(std::int64_t axis) const;  ///< negative axis counts from the end
  std::int64_t numel() const;                  ///< product of shape, 0 for empty
  std::size_t nbytes() const;                  ///< storage footprint of the elements

  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  bool defined() const { return storage_ != nullptr; }

  /// True when the strides are exactly the row-major packing of the shape, i.e.
  /// the elements are laid out linearly with no gaps. Kernels require this.
  bool is_contiguous() const;

  // --- views (no data movement) ---------------------------------------------
  /// Same data, new shape. Requires is_contiguous() and matching numel().
  /// One dimension may be -1 and is inferred.
  Tensor reshape(std::vector<std::int64_t> new_shape) const;

  /// Reorders axes by permuting shape and strides. Result is generally NOT
  /// contiguous. `perm` must be a permutation of [0, dim).
  Tensor permute(std::vector<std::int64_t> perm) const;

  /// Swaps two axes. transpose(0,1) on a 2-D tensor is the usual matrix transpose.
  Tensor transpose(std::int64_t a, std::int64_t b) const;

  /// Narrows `axis` to [start, start+length). This is the KV-cache primitive.
  Tensor slice(std::int64_t axis, std::int64_t start, std::int64_t length) const;

  // --- data movement (copies) ----------------------------------------------
  /// Returns *this if already contiguous, otherwise a packed copy.
  Tensor contiguous() const;

  /// Copies to `target` device. Returns *this if already resident there.
  Tensor to(Device target) const;

  /// Deep copy, always allocating fresh storage.
  Tensor clone() const;

  // --- raw access ----------------------------------------------------------
  /// Pointer to element zero of THIS VIEW (storage base + byte offset).
  void* data();
  const void* data() const;

  /// Typed accessor. Checks that T matches dtype() and that the tensor is on the
  /// expected device, so `t.ptr<float>(Device::CUDA)` is self-documenting at the
  /// kernel call site.
  template <typename T>
  T* ptr(Device expect);
  template <typename T>
  const T* ptr(Device expect) const;

  const std::shared_ptr<Storage>& storage() const { return storage_; }
  std::int64_t storage_offset() const { return offset_; }

  // --- debugging -----------------------------------------------------------
  /// e.g. "Tensor(shape=[2,3], strides=[3,1], dtype=f32, device=cuda, contiguous)"
  /// Implement this FIRST. Every later module is debugged through it.
  std::string to_string() const;

 private:
  /// Row-major strides for a given shape.
  static std::vector<std::int64_t> contiguous_strides(
      const std::vector<std::int64_t>& shape);

  std::shared_ptr<Storage> storage_;
  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  std::int64_t offset_ = 0;  ///< in ELEMENTS, not bytes
  DType dtype_ = DType::F32;
  Device device_ = Device::CPU;
};

}  // namespace engine
