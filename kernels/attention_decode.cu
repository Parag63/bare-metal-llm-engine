//===----------------------------------------------------------------------===//
// kernels/attention_decode.cu -- Single-token decode attention (FlashDecoding).
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {

void attention_decode(const half* q, const half* k, const half* v, half* out,
                      std::int64_t kv_len, std::int64_t n_heads, std::int64_t n_kv_heads,
                      std::int64_t head_dim, std::int64_t kv_head_stride, float scale,
                      int split_kv, cudaStream_t stream) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)kv_len;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)kv_head_stride;
  (void)scale;
  (void)split_kv;
  (void)stream;
  ENGINE_CHECK(false, "attention_decode not implemented yet (Module 4)");
}

}  // namespace engine::cuda
