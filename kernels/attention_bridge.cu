//===----------------------------------------------------------------------===//
// kernels/attention_bridge.cu -- Utility bridge kernels connecting FP32
// projections/residual streams with FP16 RoPE + head-major Attention KV cache.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda::detail {
namespace {

constexpr int kBridgeBlockThreads = 256;

__global__ void cast_f32_to_f16_kernel(const float* __restrict__ in,
                                       half* __restrict__ out, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    out[i] = __float2half(in[i]);
  }
}

__global__ void cast_f16_to_f32_kernel(const half* __restrict__ in,
                                       float* __restrict__ out, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    out[i] = __half2float(in[i]);
  }
}

__global__ void kv_cache_write_kernel(const half* __restrict__ k_new,
                                      const half* __restrict__ v_new,
                                      half* __restrict__ k_cache,
                                      half* __restrict__ v_cache, std::int64_t s_new,
                                      std::int64_t n_kv_heads, std::int64_t head_dim,
                                      std::int64_t pos_offset,
                                      std::int64_t kv_max_seq_len) {
  const std::int64_t total = s_new * n_kv_heads * head_dim;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::int64_t d = idx % head_dim;
    const std::int64_t t_h = idx / head_dim;
    const std::int64_t h = t_h % n_kv_heads;
    const std::int64_t t = t_h / n_kv_heads;
    const std::int64_t dst = (h * kv_max_seq_len + pos_offset + t) * head_dim + d;
    k_cache[dst] = k_new[idx];
    v_cache[dst] = v_new[idx];
  }
}

unsigned grid_for(std::int64_t n) {
  const std::int64_t blocks = (n + kBridgeBlockThreads - 1) / kBridgeBlockThreads;
  return static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
}

}  // namespace

void cast_f32_to_f16(const float* in, half* out, std::int64_t n, cudaStream_t stream) {
  ENGINE_CHECK(n >= 0, "cast_f32_to_f16: n must be non-negative");
  if (n == 0) {
    return;
  }
  ENGINE_CHECK(in != nullptr, "cast_f32_to_f16: in is null");
  ENGINE_CHECK(out != nullptr, "cast_f32_to_f16: out is null");
  cast_f32_to_f16_kernel<<<grid_for(n), kBridgeBlockThreads, 0, stream>>>(in, out, n);
  CUDA_CHECK_KERNEL();
}

void cast_f16_to_f32(const half* in, float* out, std::int64_t n, cudaStream_t stream) {
  ENGINE_CHECK(n >= 0, "cast_f16_to_f32: n must be non-negative");
  if (n == 0) {
    return;
  }
  ENGINE_CHECK(in != nullptr, "cast_f16_to_f32: in is null");
  ENGINE_CHECK(out != nullptr, "cast_f16_to_f32: out is null");
  cast_f16_to_f32_kernel<<<grid_for(n), kBridgeBlockThreads, 0, stream>>>(in, out, n);
  CUDA_CHECK_KERNEL();
}

void kv_cache_write(const half* k_new, const half* v_new, half* k_cache, half* v_cache,
                    std::int64_t s_new, std::int64_t n_kv_heads, std::int64_t head_dim,
                    std::int64_t pos_offset, std::int64_t kv_max_seq_len,
                    cudaStream_t stream) {
  ENGINE_CHECK(s_new >= 0, "kv_cache_write: s_new must be non-negative");
  ENGINE_CHECK(n_kv_heads > 0, "kv_cache_write: n_kv_heads must be positive");
  ENGINE_CHECK(head_dim > 0, "kv_cache_write: head_dim must be positive");
  ENGINE_CHECK(pos_offset >= 0, "kv_cache_write: pos_offset must be non-negative");
  ENGINE_CHECK(kv_max_seq_len >= pos_offset + s_new,
               "kv_cache_write: kv_max_seq_len < pos_offset + s_new");
  if (s_new == 0) {
    return;
  }
  ENGINE_CHECK(k_new != nullptr && v_new != nullptr,
               "kv_cache_write: k_new/v_new is null");
  ENGINE_CHECK(k_cache != nullptr && v_cache != nullptr,
               "kv_cache_write: k_cache/v_cache is null");

  const std::int64_t total = s_new * n_kv_heads * head_dim;
  kv_cache_write_kernel<<<grid_for(total), kBridgeBlockThreads, 0, stream>>>(
      k_new, v_new, k_cache, v_cache, s_new, n_kv_heads, head_dim, pos_offset,
      kv_max_seq_len);
  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda::detail
