//===----------------------------------------------------------------------===//
// src/cpu_ref/swiglu_cpu.cpp -- CPU reference for SwiGLU activation.
//
// Computes out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
// Uses double precision in intermediate transcendental computations.
//===----------------------------------------------------------------------===//

#include <engine/cpu_ref.hpp>
#include <engine/check.hpp>
#include <cmath>

namespace engine::cpu {

void swiglu(const float* gate, const float* up, float* out, std::int64_t n) {
  ENGINE_CHECK(gate != nullptr, "swiglu: gate is null");
  ENGINE_CHECK(up != nullptr, "swiglu: up is null");
  ENGINE_CHECK(out != nullptr, "swiglu: out is null");
  ENGINE_CHECK(n >= 0, "swiglu: n must be non-negative");

  for (std::int64_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(gate[i]);
    const double u = static_cast<double>(up[i]);
    const double silu = g / (1.0 + std::exp(-g));
    out[i] = static_cast<float>(silu * u);
  }
}

void swiglu_fp16(const half* gate, const half* up, half* out, std::int64_t n) {
  ENGINE_CHECK(gate != nullptr, "swiglu_fp16: gate is null");
  ENGINE_CHECK(up != nullptr, "swiglu_fp16: up is null");
  ENGINE_CHECK(out != nullptr, "swiglu_fp16: out is null");
  ENGINE_CHECK(n >= 0, "swiglu_fp16: n must be non-negative");

  for (std::int64_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(half_to_float(gate[i]));
    const double u = static_cast<double>(half_to_float(up[i]));
    const double silu = g / (1.0 + std::exp(-g));
    out[i] = float_to_half(static_cast<float>(silu * u));
  }
}

}  // namespace engine::cpu
