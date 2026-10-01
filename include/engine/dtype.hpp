#pragma once
//===----------------------------------------------------------------------===//
// engine/dtype.hpp -- data types the engine can store.
//
// Objective 1 requires FP32, FP16, INT8 and INT4 support. INT4 is the interesting
// one: it is SUB-BYTE, so "how many bytes is one element" has no integer answer.
// That is why the API below is expressed in *bits* per element, with a separate
// helper for the block-quantised layouts that GGUF actually uses.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>

#include <cstddef>
#include <cstdint>
#include <ostream>

namespace engine {

enum class DType : std::uint8_t {
  F32 = 0,   ///< IEEE-754 single precision. The reference type; every kernel starts here.
  F16 = 1,   ///< IEEE-754 half. Native + tensor-core accelerated on Ada (sm_89).
  BF16 = 2,  ///< bfloat16. Same 8 exponent bits as FP32 but only 7 mantissa bits, so it
             ///< has FP32's dynamic range and cannot overflow where FP32 would not.
             ///< Natively supported on the RTX 4090 (Ampere sm_80 and later). Generally
             ///< the better default than F16 for transformer weights, since FP16's
             ///< narrow exponent is what forces loss-scaling tricks.
  I32 = 3,   ///< 32-bit signed integer (indices, token ids).
  I8 = 4,    ///< 8-bit signed integer. Quantisation target, tensor-core capable.
  U8 = 5,    ///< 8-bit unsigned, used as the storage unit for packed sub-byte data.
  I4 = 6,    ///< 4-bit signed, PACKED two-per-byte. Never addressable directly.
  COUNT = 7
};

/// Bits occupied by a single element. This is the honest primitive -- prefer it.
constexpr int dtype_bits(DType dt) {
  switch (dt) {
    case DType::F32:
      return 32;
    case DType::F16:
      return 16;
    case DType::BF16:
      return 16;
    case DType::I32:
      return 32;
    case DType::I8:
      return 8;
    case DType::U8:
      return 8;
    case DType::I4:
      return 4;
    default:
      return 0;
  }
}

/// True for types where one element does not occupy a whole number of bytes.
constexpr bool dtype_is_sub_byte(DType dt) { return dtype_bits(dt) < 8; }

constexpr bool dtype_is_float(DType dt) {
  return dt == DType::F32 || dt == DType::F16 || dt == DType::BF16;
}

constexpr bool dtype_is_quantised(DType dt) { return dt == DType::I8 || dt == DType::I4; }

/// Bytes per element. Throws for sub-byte types -- by design, so that code which
/// assumes byte-addressability cannot silently mis-size an INT4 buffer.
/// For packed types use dtype_storage_bytes() instead.
inline std::size_t dtype_size(DType dt) {
  ENGINE_CHECK(!dtype_is_sub_byte(dt),
               "dtype_size() is undefined for sub-byte types; use dtype_storage_bytes()");
  const int bits = dtype_bits(dt);
  ENGINE_CHECK(bits > 0, "unknown DType passed to dtype_size()");
  return static_cast<std::size_t>(bits) / 8u;
}

/// Bytes needed to store `n` elements of type `dt`, rounding sub-byte types up to
/// a whole byte. Works for every type including I4.
inline std::size_t dtype_storage_bytes(DType dt, std::size_t n) {
  const int bits = dtype_bits(dt);
  ENGINE_CHECK(bits > 0, "unknown DType passed to dtype_storage_bytes()");
  const std::size_t total_bits = n * static_cast<std::size_t>(bits);
  return (total_bits + 7u) / 8u;  // ceil-divide to whole bytes
}

const char* dtype_name(DType dt);

/// Streamable so that DType prints as "f32" rather than as an integer, and so that
/// generic code (test assertions, log lines, Tensor::to_string) can just use <<.
/// Found by argument-dependent lookup, so it works from any namespace.
inline std::ostream& operator<<(std::ostream& os, DType dt) {
  return os << dtype_name(dt);
}

//===----------------------------------------------------------------------===//
// Block quantisation (Objective 5, May 2027 -- declared early so the Tensor
// layout does not have to be redesigned when you get there)
//
// Real quantised formats do not store one scale for a whole tensor; the dynamic
// range across a 4096x4096 weight matrix is far too wide for that. GGUF's Q4_0
// instead splits the tensor into fixed-size BLOCKS of 32 elements and stores one
// FP16 scale per block:
//
//     struct block_q4_0 { half scale; uint8_t nibbles[16]; };  // 32 weights -> 18 bytes
//
// That is 4.5 bits/weight, not 4.0 -- the scale is real overhead you must account
// for when you report compression ratios in the final benchmark table.
//===----------------------------------------------------------------------===//

/// Elements per quantisation block. 32 matches GGUF's Q4_0/Q8_0 families.
constexpr int kQuantBlockSize = 32;

/// Bytes per block for a quantised type, including the FP16 scale.
inline std::size_t quant_block_bytes(DType dt) {
  ENGINE_CHECK(dtype_is_quantised(dt), "quant_block_bytes() requires a quantised DType");
  const std::size_t payload =
      dtype_storage_bytes(dt, static_cast<std::size_t>(kQuantBlockSize));
  return payload + sizeof(std::uint16_t);  // + one FP16 scale
}

/// Effective bits per weight *including* scale overhead. Use this number in the
/// report, not the nominal bit width.
inline double quant_effective_bits(DType dt) {
  return 8.0 * static_cast<double>(quant_block_bytes(dt)) /
         static_cast<double>(kQuantBlockSize);
}

}  // namespace engine
