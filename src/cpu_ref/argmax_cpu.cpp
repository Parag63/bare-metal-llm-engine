//===----------------------------------------------------------------------===//
// src/cpu_ref/argmax_cpu.cpp -- CPU reference for argmax / greedy sampling.
//
// Finds index of the maximum value in logits[0..vocab_size).
// Tie-breaking: picks the lowest index.
//===----------------------------------------------------------------------===//

#include <engine/cpu_ref.hpp>
#include <engine/check.hpp>
#include <string>

namespace engine::cpu {

std::int32_t argmax(const float* logits, std::int64_t vocab_size) {
  ENGINE_CHECK(logits != nullptr, "argmax: logits is null");
  ENGINE_CHECK(vocab_size > 0, "argmax: vocab_size must be positive");

  float max_val = logits[0];
  std::int32_t max_idx = 0;

  for (std::int64_t i = 1; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = static_cast<std::int32_t>(i);
    }
  }

  return max_idx;
}

std::int32_t argmax_fp16(const half* logits, std::int64_t vocab_size) {
  ENGINE_CHECK(logits != nullptr, "argmax_fp16: logits is null");
  ENGINE_CHECK(vocab_size > 0, "argmax_fp16: vocab_size must be positive");

  float max_val = half_to_float(logits[0]);
  std::int32_t max_idx = 0;

  for (std::int64_t i = 1; i < vocab_size; ++i) {
    float val = half_to_float(logits[i]);
    if (val > max_val) {
      max_val = val;
      max_idx = static_cast<std::int32_t>(i);
    }
  }

  return max_idx;
}

}  // namespace engine::cpu
