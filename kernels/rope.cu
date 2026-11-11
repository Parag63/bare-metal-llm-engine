//===----------------------------------------------------------------------===//
// kernels/rope.cu -- In-place Rotary Position Embedding (RoPE) kernel.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {

void rope(half* q, half* k, const float* cos_table, const float* sin_table,
          std::int64_t s, std::int64_t n_heads, std::int64_t n_kv_heads,
          std::int64_t head_dim, std::int64_t pos_offset, RopeConvention convention,
          cudaStream_t stream) {
  (void)q;
  (void)k;
  (void)cos_table;
  (void)sin_table;
  (void)s;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)pos_offset;
  (void)convention;
  (void)stream;
  ENGINE_CHECK(false, "rope not implemented yet (Module 4)");
}

}  // namespace engine::cuda
