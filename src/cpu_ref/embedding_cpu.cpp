//===----------------------------------------------------------------------===//
// src/cpu_ref/embedding_cpu.cpp -- CPU reference for embedding lookup.
//
// table: [vocab_size, hidden_dim], row-major
// input_ids: [num_tokens]
// out: [num_tokens, hidden_dim]
//
// For each token t in [0, num_tokens):
//   id = input_ids[t]
//   assert id >= 0 && id < vocab_size
//   copy table[id * hidden_dim ..] to out[t * hidden_dim ..]
//===----------------------------------------------------------------------===//

#include <engine/cpu_ref.hpp>
#include <engine/check.hpp>
#include <cstring>
#include <string>

namespace engine::cpu {

void embedding(const float* table, const std::int32_t* input_ids, float* out,
               std::int64_t num_tokens, std::int64_t hidden_dim,
               std::int64_t vocab_size) {
  ENGINE_CHECK(table != nullptr, "embedding: table is null");
  ENGINE_CHECK(input_ids != nullptr, "embedding: input_ids is null");
  ENGINE_CHECK(out != nullptr, "embedding: out is null");
  ENGINE_CHECK(num_tokens >= 0, "embedding: num_tokens must be non-negative");
  ENGINE_CHECK(hidden_dim >= 0, "embedding: hidden_dim must be non-negative");
  ENGINE_CHECK(vocab_size > 0, "embedding: vocab_size must be positive");

  for (std::int64_t t = 0; t < num_tokens; ++t) {
    const std::int32_t id = input_ids[t];
    ENGINE_CHECK(id >= 0 && id < vocab_size, "embedding: input_id " + std::to_string(id) +
                                                 " out of bounds [0, " +
                                                 std::to_string(vocab_size) + ")");
    std::memcpy(out + t * hidden_dim, table + static_cast<std::int64_t>(id) * hidden_dim,
                static_cast<std::size_t>(hidden_dim) * sizeof(float));
  }
}

void embedding_fp16(const half* table, const std::int32_t* input_ids, half* out,
                    std::int64_t num_tokens, std::int64_t hidden_dim,
                    std::int64_t vocab_size) {
  ENGINE_CHECK(table != nullptr, "embedding_fp16: table is null");
  ENGINE_CHECK(input_ids != nullptr, "embedding_fp16: input_ids is null");
  ENGINE_CHECK(out != nullptr, "embedding_fp16: out is null");
  ENGINE_CHECK(num_tokens >= 0, "embedding_fp16: num_tokens must be non-negative");
  ENGINE_CHECK(hidden_dim >= 0, "embedding_fp16: hidden_dim must be non-negative");
  ENGINE_CHECK(vocab_size > 0, "embedding_fp16: vocab_size must be positive");

  for (std::int64_t t = 0; t < num_tokens; ++t) {
    const std::int32_t id = input_ids[t];
    ENGINE_CHECK(id >= 0 && id < vocab_size,
                 "embedding_fp16: input_id " + std::to_string(id) +
                     " out of bounds [0, " + std::to_string(vocab_size) + ")");
    std::memcpy(out + t * hidden_dim, table + static_cast<std::int64_t>(id) * hidden_dim,
                static_cast<std::size_t>(hidden_dim) * sizeof(half));
  }
}

}  // namespace engine::cpu
