#pragma once
//===----------------------------------------------------------------------===//
// engine/half.hpp -- Half-precision floating point type (IEEE 754 binary16).
//
// Bridges CUDA __half (when CUDA is enabled) and a portable host-side half type
// (for CPU-only builds), providing transparent conversion helpers and operators.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#include <cstdint>
#include <cstring>
#include <ostream>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define ENGINE_HOST_DEVICE __host__ __device__
#else
#define ENGINE_HOST_DEVICE
#endif

#if ENGINE_HAS_CUDA
#include <cuda_fp16.h>
#endif

namespace engine {

#if ENGINE_HAS_CUDA
using half = __half;

ENGINE_HOST_DEVICE inline float half_to_float(half h) { return __half2float(h); }

ENGINE_HOST_DEVICE inline half float_to_half(float f) { return __float2half(f); }

#else

/// IEEE-754 binary16 half precision structure for CPU-only toolchains.
struct half {
  std::uint16_t u = 0;

  half() = default;
  constexpr explicit half(std::uint16_t raw) : u(raw) {}
};

inline std::uint16_t float_to_half_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(float));
  std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xff) - 127 + 15;
  std::uint32_t mant = x & 0x007fffffu;

  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    mant = (mant | 0x00800000u) >> static_cast<std::uint32_t>(1 - exp);
    return static_cast<std::uint16_t>(sign | (mant >> 13));
  } else if (exp >= 31) {
    if (exp == 143 && mant != 0) return static_cast<std::uint16_t>(sign | 0x7e00u);
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                    (mant >> 13));
}

inline float half_bits_to_float(std::uint16_t h) {
  std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
  std::int32_t exp = static_cast<std::int32_t>((h >> 10) & 0x1f);
  std::uint32_t mant = static_cast<std::uint32_t>(h & 0x03ffu);
  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03ffu;
      out = sign | (static_cast<std::uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | (static_cast<std::uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(float));
  return f;
}

inline float half_to_float(half h) { return half_bits_to_float(h.u); }

inline half float_to_half(float f) { return half(float_to_half_bits(f)); }

inline bool operator==(half a, half b) { return a.u == b.u; }
inline bool operator!=(half a, half b) { return a.u != b.u; }
inline bool operator<(half a, half b) { return half_to_float(a) < half_to_float(b); }
inline bool operator<=(half a, half b) { return half_to_float(a) <= half_to_float(b); }
inline bool operator>(half a, half b) { return half_to_float(a) > half_to_float(b); }
inline bool operator>=(half a, half b) { return half_to_float(a) >= half_to_float(b); }

#endif

inline std::ostream& operator<<(std::ostream& os, half h) {
  return os << half_to_float(h);
}

}  // namespace engine
