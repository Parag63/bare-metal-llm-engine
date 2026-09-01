//===----------------------------------------------------------------------===//
// src/tensor.cpp -- Module 1 (Objective 1). IMPLEMENTED.
//
// Storage is implemented: it is allocation plumbing, and the CUDA-guarded
// device dispatch is fiddly boilerplate rather than something you learn from.
//
// The Tensor methods below are the actual idea of Module 1: shape and strides
// are metadata, and most tensor operations are metadata edits rather than data
// movement.
//
// Implementation order (each step unblocked by the one before it):
//   1. contiguous_strides, numel, nbytes, size
//   2. to_string
//   3. constructor, zeros
//   4. is_contiguous
//   5. data, ptr<T>
//   6. reshape
//   7. permute, transpose
//   8. slice
//   9. contiguous, clone, to
//  10. from_blob
//===----------------------------------------------------------------------===//

#include <engine/tensor.hpp>

#include <engine/check.hpp>
#include <engine/config.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace engine {

const char* device_name(Device d) {
  switch (d) {
    case Device::CPU: return "cpu";
    case Device::CUDA: return "cuda";
    default: return "<invalid>";
  }
}

//===----------------------------------------------------------------------===//
// Storage -- IMPLEMENTED. Read it, but you do not need to change it.
//===----------------------------------------------------------------------===//

Storage::Storage(std::size_t bytes, Device device) : bytes_(bytes), device_(device) {
  if (bytes == 0) {
    return;  // zero-byte storage is legal and holds no pointer
  }

  if (device == Device::CPU) {
    ptr_ = std::malloc(bytes);
    ENGINE_CHECK(ptr_ != nullptr, "host allocation failed");
  } else {
#if ENGINE_HAS_CUDA
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));
#else
    // Reaching here means someone asked for a CUDA tensor in a CPU-only build.
    // Fail with a message that says what to do, not just "unsupported".
    ENGINE_CHECK(false,
                 "requested Device::CUDA but this build has no CUDA support. "
                 "Build on the RTX 4090 machine, or pass Device::CPU here.");
#endif
  }
}

Storage::Storage(void* data, std::size_t bytes, Device device, BorrowTag)
    : ptr_(data), bytes_(bytes), device_(device), owns_(false) {
  // Non-owning: the caller guarantees the pointer outlives this Storage.
  // No allocation happens here.
}

Storage::~Storage() {
  if (!owns_ || ptr_ == nullptr) return;

  if (device_ == Device::CPU) {
    std::free(ptr_);
  } else {
#if ENGINE_HAS_CUDA
    // Destructors must not throw, so the error is deliberately ignored.
    cudaFree(ptr_);
#endif
  }
}

//===----------------------------------------------------------------------===//
// Tensor -- IMPLEMENTED.
//===----------------------------------------------------------------------===//

// ---- Step 1: pure arithmetic, no allocation --------------------------------

std::vector<std::int64_t> Tensor::contiguous_strides(
    const std::vector<std::int64_t>& shape) {
  // Row-major strides: the last axis has stride 1, and walking backwards each
  // stride is the product of all later dimensions.
  //   [2,3,4] -> [12, 4, 1]
  //   []      -> []  (a 0-d scalar tensor has no strides)
  if (shape.empty()) return {};

  std::vector<std::int64_t> strides(shape.size());
  strides.back() = 1;
  for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] =
        strides[static_cast<std::size_t>(i + 1)] *
        shape[static_cast<std::size_t>(i + 1)];
  }
  return strides;
}

std::int64_t Tensor::numel() const {
  // Product of shape_. Convention:
  //   - 0-d tensor (shape {}): empty product is 1, so numel = 1.
  //   - Undefined tensor (no storage): numel = 0.
  //   - Any axis of size 0 makes the whole tensor empty.
  if (!defined()) return 0;
  return std::accumulate(shape_.begin(), shape_.end(), std::int64_t{1},
                         std::multiplies<std::int64_t>());
}

std::size_t Tensor::nbytes() const {
  // Storage footprint of numel() elements. Uses dtype_storage_bytes() rather
  // than dtype_size() so that sub-byte types (INT4) are handled correctly.
  const std::int64_t n = numel();
  if (n <= 0) return 0;
  return dtype_storage_bytes(dtype_, static_cast<std::size_t>(n));
}

std::int64_t Tensor::size(std::int64_t axis) const {
  // Supports negative indexing: -1 is the last axis.
  if (axis < 0) axis += dim();
  ENGINE_CHECK(axis >= 0 && axis < dim(),
               "Tensor::size: axis " + std::to_string(axis) +
                   " out of range for tensor with dim=" + std::to_string(dim()));
  return shape_[static_cast<std::size_t>(axis)];
}

// ---- Step 2: to_string (implement early -- you debug everything through it) -

std::string Tensor::to_string() const {
  if (!defined()) return "Tensor(undefined)";

  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape_[i];
  }
  oss << "], strides=[";
  for (std::size_t i = 0; i < strides_.size(); ++i) {
    if (i > 0) oss << ",";
    oss << strides_[i];
  }
  oss << "], dtype=" << dtype_name(dtype_)
      << ", device=" << device_name(device_)
      << ", " << (is_contiguous() ? "contiguous" : "non-contiguous")
      << ")";
  return oss.str();
}

// ---- Step 3: allocation and zeros ------------------------------------------

Tensor::Tensor(std::vector<std::int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
  // Validate: every dimension must be >= 0.
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    ENGINE_CHECK(shape_[i] >= 0,
                 "Tensor: dimension " + std::to_string(i) + " is " +
                     std::to_string(shape_[i]) + ", must be >= 0");
  }

  strides_ = contiguous_strides(shape_);

  // Compute byte size and allocate. The product of shape may be zero (e.g.
  // shape {4, 0, 8}), in which case we still create a storage of 0 bytes.
  const std::int64_t n = std::accumulate(shape_.begin(), shape_.end(),
                                         std::int64_t{1},
                                         std::multiplies<std::int64_t>());
  const std::size_t bytes = (n > 0) ? dtype_storage_bytes(dtype_, static_cast<std::size_t>(n))
                                    : 0;
  storage_ = std::make_shared<Storage>(bytes, device_);
  offset_ = 0;
}

Tensor Tensor::zeros(std::vector<std::int64_t> shape, DType dtype, Device device) {
  Tensor t(std::move(shape), dtype, device);
  const std::size_t bytes = t.nbytes();
  if (bytes == 0) return t;

  if (device == Device::CPU) {
    std::memset(t.storage_->data(), 0, bytes);
  } else {
#if ENGINE_HAS_CUDA
    // cudaMemset takes a byte value. All-zero bits is 0.0f in IEEE 754, so
    // this is correct for floats. It would NOT work for filling with 1.0f.
    CUDA_CHECK(cudaMemset(t.storage_->data(), 0, bytes));
#else
    ENGINE_CHECK(false,
                 "Tensor::zeros on Device::CUDA in a CPU-only build");
#endif
  }
  return t;
}

// ---- Step 4: is_contiguous --------------------------------------------------

bool Tensor::is_contiguous() const {
  // True when the elements are packed in row-major order with no gaps.
  //
  // Design decision on degenerate cases: any axis of size 0 or 1 makes its
  // stride irrelevant (you can never take more than zero steps along it), so
  // we skip those axes when comparing against the expected contiguous stride.
  // A strict comparison would report [1,3]-with-strides-[42,1] as
  // non-contiguous, but that tensor IS contiguous: its elements are at
  // offsets 0, 1, 2 regardless of the stride on the size-1 axis.
  if (shape_.empty()) return true;  // 0-d scalar is contiguous

  std::int64_t expected = 1;
  for (std::int64_t i = dim() - 1; i >= 0; --i) {
    const auto idx = static_cast<std::size_t>(i);
    if (shape_[idx] == 0) return true;  // empty tensor is contiguous by convention
    if (shape_[idx] > 1) {
      if (strides_[idx] != expected) return false;
    }
    expected *= shape_[idx];
  }
  return true;
}

// ---- Step 5: data() and ptr<T>() -------------------------------------------

void* Tensor::data() {
  ENGINE_CHECK(defined(), "Tensor::data() called on undefined tensor");
  // offset_ is in ELEMENTS. Cast to char* for byte arithmetic, then advance
  // by offset_ * element_size_in_bytes. For sub-byte types this is approximate
  // (INT4 at odd offsets), but that case is handled at the kernel level, not here.
  auto* base = static_cast<char*>(storage_->data());
  if (dtype_is_sub_byte(dtype_)) {
    // For sub-byte types, offset is in number of sub-byte elements. Convert to
    // bytes: offset_ * bits / 8. We require byte-aligned offsets for sub-byte.
    const std::size_t bit_offset =
        static_cast<std::size_t>(offset_) * static_cast<std::size_t>(dtype_bits(dtype_));
    return base + bit_offset / 8;
  }
  return base + offset_ * static_cast<std::int64_t>(dtype_size(dtype_));
}

const void* Tensor::data() const {
  // Implement in terms of the non-const overload to avoid duplicating the
  // offset arithmetic.
  return const_cast<Tensor*>(this)->data();
}

// ---- Step 6: reshape -------------------------------------------------------

Tensor Tensor::reshape(std::vector<std::int64_t> new_shape) const {
  ENGINE_CHECK(defined(), "Tensor::reshape called on undefined tensor");
  ENGINE_CHECK(is_contiguous(),
               "Tensor::reshape requires a contiguous tensor. "
               "Call .contiguous() first to pack the data.");

  // Resolve at most one -1 dimension.
  const std::int64_t n = numel();
  std::int64_t infer_idx = -1;
  std::int64_t known_product = 1;
  for (std::size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      ENGINE_CHECK(infer_idx == -1,
                   "Tensor::reshape: at most one dimension may be -1");
      infer_idx = static_cast<std::int64_t>(i);
    } else {
      ENGINE_CHECK(new_shape[i] >= 0,
                   "Tensor::reshape: negative dimension " +
                       std::to_string(new_shape[i]));
      known_product *= new_shape[i];
    }
  }

  if (infer_idx >= 0) {
    ENGINE_CHECK(known_product > 0 && n % known_product == 0,
                 "Tensor::reshape: cannot infer dimension for shape");
    new_shape[static_cast<std::size_t>(infer_idx)] = n / known_product;
  }

  // Verify the element count matches.
  const std::int64_t new_n = std::accumulate(
      new_shape.begin(), new_shape.end(), std::int64_t{1},
      std::multiplies<std::int64_t>());
  ENGINE_CHECK(new_n == n,
               "Tensor::reshape: cannot reshape " + std::to_string(n) +
                   " elements into " + std::to_string(new_n));

  // Build the result: same storage, new shape, freshly computed contiguous strides.
  Tensor out;
  out.storage_ = storage_;
  out.shape_ = std::move(new_shape);
  out.strides_ = contiguous_strides(out.shape_);
  out.offset_ = offset_;
  out.dtype_ = dtype_;
  out.device_ = device_;
  return out;
}

// ---- Step 7: permute and transpose -----------------------------------------

Tensor Tensor::permute(std::vector<std::int64_t> perm) const {
  ENGINE_CHECK(defined(), "Tensor::permute called on undefined tensor");
  ENGINE_CHECK(static_cast<std::int64_t>(perm.size()) == dim(),
               "Tensor::permute: perm length " + std::to_string(perm.size()) +
                   " != dim " + std::to_string(dim()));

  // Validate: perm must be a permutation of [0, dim). Every index exactly once.
  std::vector<bool> seen(static_cast<std::size_t>(dim()), false);
  for (std::size_t i = 0; i < perm.size(); ++i) {
    ENGINE_CHECK(perm[i] >= 0 && perm[i] < dim(),
                 "Tensor::permute: axis " + std::to_string(perm[i]) +
                     " out of range for dim=" + std::to_string(dim()));
    ENGINE_CHECK(!seen[static_cast<std::size_t>(perm[i])],
                 "Tensor::permute: axis " + std::to_string(perm[i]) +
                     " appears more than once");
    seen[static_cast<std::size_t>(perm[i])] = true;
  }

  Tensor out;
  out.storage_ = storage_;
  out.shape_.resize(static_cast<std::size_t>(dim()));
  out.strides_.resize(static_cast<std::size_t>(dim()));
  for (std::size_t i = 0; i < perm.size(); ++i) {
    out.shape_[i] = shape_[static_cast<std::size_t>(perm[i])];
    out.strides_[i] = strides_[static_cast<std::size_t>(perm[i])];
  }
  out.offset_ = offset_;
  out.dtype_ = dtype_;
  out.device_ = device_;
  return out;
}

Tensor Tensor::transpose(std::int64_t a, std::int64_t b) const {
  // Normalise negative axes.
  if (a < 0) a += dim();
  if (b < 0) b += dim();
  ENGINE_CHECK(a >= 0 && a < dim(),
               "Tensor::transpose: axis " + std::to_string(a) +
                   " out of range for dim=" + std::to_string(dim()));
  ENGINE_CHECK(b >= 0 && b < dim(),
               "Tensor::transpose: axis " + std::to_string(b) +
                   " out of range for dim=" + std::to_string(dim()));

  // Build the identity permutation, then swap a and b.
  std::vector<std::int64_t> perm(static_cast<std::size_t>(dim()));
  std::iota(perm.begin(), perm.end(), std::int64_t{0});
  std::swap(perm[static_cast<std::size_t>(a)],
            perm[static_cast<std::size_t>(b)]);
  return permute(perm);
}

// ---- Step 8: slice ---------------------------------------------------------

Tensor Tensor::slice(std::int64_t axis, std::int64_t start,
                     std::int64_t length) const {
  ENGINE_CHECK(defined(), "Tensor::slice called on undefined tensor");

  // Normalise negative axis.
  if (axis < 0) axis += dim();
  ENGINE_CHECK(axis >= 0 && axis < dim(),
               "Tensor::slice: axis " + std::to_string(axis) +
                   " out of range for dim=" + std::to_string(dim()));
  ENGINE_CHECK(start >= 0,
               "Tensor::slice: start must be >= 0, got " + std::to_string(start));
  ENGINE_CHECK(length >= 0,
               "Tensor::slice: length must be >= 0, got " + std::to_string(length));
  ENGINE_CHECK(start + length <= shape_[static_cast<std::size_t>(axis)],
               "Tensor::slice: start(" + std::to_string(start) + ") + length(" +
                   std::to_string(length) + ") > axis size(" +
                   std::to_string(shape_[static_cast<std::size_t>(axis)]) + ")");

  Tensor out;
  out.storage_ = storage_;
  out.shape_ = shape_;
  out.strides_ = strides_;
  out.shape_[static_cast<std::size_t>(axis)] = length;
  out.offset_ = offset_ + start * strides_[static_cast<std::size_t>(axis)];
  out.dtype_ = dtype_;
  out.device_ = device_;
  return out;
}

// ---- Step 9: the copying operations ----------------------------------------

Tensor Tensor::contiguous() const {
  ENGINE_CHECK(defined(), "Tensor::contiguous called on undefined tensor");
  if (is_contiguous()) return *this;  // cheap, very common

  // General N-dimensional strided copy via an odometer.
  // Allocate a fresh contiguous tensor with the same shape.
  Tensor out(shape_, dtype_, device_);

  ENGINE_CHECK(device_ == Device::CPU,
               "Tensor::contiguous on a non-contiguous CUDA tensor is not supported. "
               "Make views contiguous on CPU before transferring to CUDA.");

  const std::int64_t n = numel();
  if (n == 0) return out;

  const std::size_t elem_size = dtype_size(dtype_);
  const auto* src_base = static_cast<const char*>(data());
  auto* dst = static_cast<char*>(out.data());

  // The odometer: one counter per axis, walking in row-major order.
  const std::int64_t rank = dim();
  std::vector<std::int64_t> idx(static_cast<std::size_t>(rank), 0);

  for (std::int64_t flat = 0; flat < n; ++flat) {
    // Compute the source offset from the multi-dimensional index and strides.
    std::int64_t src_offset = 0;
    for (std::int64_t d = 0; d < rank; ++d) {
      src_offset += idx[static_cast<std::size_t>(d)] *
                    strides_[static_cast<std::size_t>(d)];
    }

    std::memcpy(dst + flat * static_cast<std::int64_t>(elem_size),
                src_base + src_offset * static_cast<std::int64_t>(elem_size),
                elem_size);

    // Increment the odometer: carry from the last axis backwards.
    for (std::int64_t d = rank - 1; d >= 0; --d) {
      auto didx = static_cast<std::size_t>(d);
      ++idx[didx];
      if (idx[didx] < shape_[didx]) break;
      idx[didx] = 0;
    }
  }

  return out;
}

Tensor Tensor::clone() const {
  ENGINE_CHECK(defined(), "Tensor::clone called on undefined tensor");

  // Always allocate fresh storage. Unlike contiguous(), clone() never returns *this.
  // First make contiguous if needed, then copy.
  Tensor src = contiguous();
  Tensor out(src.shape_, src.dtype_, src.device_);

  const std::size_t bytes = src.nbytes();
  if (bytes == 0) return out;

  if (src.device_ == Device::CPU) {
    std::memcpy(out.storage_->data(), src.data(), bytes);
  } else {
#if ENGINE_HAS_CUDA
    CUDA_CHECK(cudaMemcpy(out.storage_->data(), src.data(), bytes,
                          cudaMemcpyDeviceToDevice));
#endif
  }
  return out;
}

Tensor Tensor::to(Device target) const {
  ENGINE_CHECK(defined(), "Tensor::to called on undefined tensor");

  // No-op when already on the target device.
  if (device_ == target) return *this;

  // Require contiguity. A strided cross-device copy is a trap: the GPU-side
  // buffer would have gaps that no kernel expects.
  ENGINE_CHECK(is_contiguous(),
               "Tensor::to requires a contiguous tensor. "
               "Call .contiguous() first to pack the data.");

  Tensor out(shape_, dtype_, target);
  const std::size_t bytes = nbytes();
  if (bytes == 0) return out;

#if ENGINE_HAS_CUDA
  cudaMemcpyKind kind;
  if (device_ == Device::CPU && target == Device::CUDA) {
    kind = cudaMemcpyHostToDevice;
  } else if (device_ == Device::CUDA && target == Device::CPU) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyDeviceToDevice;
  }
  CUDA_CHECK(cudaMemcpy(out.storage_->data(), data(), bytes, kind));
#else
  ENGINE_CHECK(false,
               "Tensor::to(Device::CUDA) in a CPU-only build. "
               "Build on the RTX 4090 machine, or stay on Device::CPU.");
#endif
  return out;
}

// ---- Step 10: from_blob (non-owning storage) -------------------------------

Tensor Tensor::from_blob(void* data, std::vector<std::int64_t> shape, DType dtype,
                         Device device) {
  ENGINE_CHECK(data != nullptr || shape.empty(),
               "Tensor::from_blob: null data with non-empty shape");

  // Validate dimensions.
  for (std::size_t i = 0; i < shape.size(); ++i) {
    ENGINE_CHECK(shape[i] >= 0,
                 "Tensor::from_blob: dimension " + std::to_string(i) + " is " +
                     std::to_string(shape[i]) + ", must be >= 0");
  }

  const std::int64_t n = std::accumulate(
      shape.begin(), shape.end(), std::int64_t{1},
      std::multiplies<std::int64_t>());
  const std::size_t bytes = (n > 0) ? dtype_storage_bytes(dtype, static_cast<std::size_t>(n))
                                    : 0;

  // Create a non-owning Storage via BorrowTag. The caller guarantees the pointer
  // outlives the tensor. This is what lets ~50 weight tensors alias one mmap'd
  // GGUF region without copying.
  auto storage = std::make_shared<Storage>(data, bytes, device, BorrowTag{});

  Tensor t;
  t.storage_ = std::move(storage);
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  t.offset_ = 0;
  t.dtype_ = dtype;
  t.device_ = device;
  return t;
}

// ---- ptr<T> ----------------------------------------------------------------

template <typename T>
T* Tensor::ptr(Device expect) {
  ENGINE_CHECK(device_ == expect,
               "Tensor::ptr: expected device " + std::string(device_name(expect)) +
                   " but tensor is on " + std::string(device_name(device_)));

  // Check that sizeof(T) matches the element size. For sub-byte types this
  // check is not meaningful, so skip it.
  if (!dtype_is_sub_byte(dtype_)) {
    ENGINE_CHECK(sizeof(T) == dtype_size(dtype_),
                 "Tensor::ptr: sizeof(T)=" + std::to_string(sizeof(T)) +
                     " does not match dtype element size " +
                     std::to_string(dtype_size(dtype_)));
  }

  return static_cast<T*>(this->data());
}

template float* Tensor::ptr<float>(Device);
template int* Tensor::ptr<int>(Device);
template unsigned char* Tensor::ptr<unsigned char>(Device);

}  // namespace engine
