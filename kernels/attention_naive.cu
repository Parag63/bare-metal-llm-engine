//===----------------------------------------------------------------------===//
// kernels/attention_naive.cu -- Naive FP16 multi-head / GQA attention baseline.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

namespace engine::cuda {

void attention_naive(const half* q, const half* k, const half* v, half* out,
                     std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                     std::int64_t n_kv_heads, std::int64_t head_dim,
                     std::int64_t kv_head_stride, float scale, std::int64_t q_pos_offset,
                     bool is_causal, float* out_lse, cudaStream_t stream) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)s_q;
  (void)s_kv;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)kv_head_stride;
  (void)scale;
  (void)q_pos_offset;
  (void)is_causal;
  (void)out_lse;
  (void)stream;
  ENGINE_CHECK(false, "attention_naive not implemented yet (Module 4)");
}

}  // namespace engine::cuda
