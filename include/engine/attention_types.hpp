//===----------------------------------------------------------------------===//
// include/engine/attention_types.hpp -- Shared types for attention and RoPE.
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>

namespace engine {

/// Pairing convention for Rotary Position Embedding (RoPE).
enum class RopeConvention : std::uint8_t {
  /// Pairs (x[d], x[d + head_dim / 2]) -- Llama / HuggingFace convention.
  RotateHalf = 0,
  /// Pairs (x[2 * d], x[2 * d + 1]) -- original Su et al. / GGUF convention.
  AdjacentPair = 1,
};

}  // namespace engine
