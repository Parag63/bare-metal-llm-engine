//===----------------------------------------------------------------------===//
// kernels/rope.cu -- In-place Rotary Position Embedding (RoPE) kernel.
//
// Applies rotary position embeddings in place to token-major Q [s, n_heads, D]
// and K [s, n_kv_heads, D] in a single fused launch using precomputed FP32
// [max_seq, D/2] cos/sin tables (ADR-0012). Rotations execute in FP32 and
// round back to FP16 on store.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {
namespace {

constexpr int kRopeBlockSize = 256;

template <int kHeadDim, bool kRotateHalf>
__global__ void rope_kernel(half* __restrict__ q, half* __restrict__ k,
                            const float* __restrict__ cos_table,
                            const float* __restrict__ sin_table, std::int64_t s,
                            std::int64_t n_heads, std::int64_t n_kv_heads,
                            std::int64_t pos_offset) {
  constexpr int kHalfD = kHeadDim / 2;
  const std::int64_t total_heads = n_heads + n_kv_heads;
  const std::int64_t total_pairs = s * total_heads * kHalfD;
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

  for (std::int64_t idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_pairs; idx += stride) {
    const int i = static_cast<int>(idx & (kHalfD - 1));
    const std::int64_t head_linear = idx / kHalfD;
    const std::int64_t h_slot = head_linear % total_heads;
    const std::int64_t t = head_linear / total_heads;
    const std::int64_t pos = pos_offset + t;

    const std::int64_t cs_idx = pos * kHalfD + i;
    const float c = cos_table[cs_idx];
    const float sn = sin_table[cs_idx];

    half* vec = (h_slot < n_heads)
                    ? (q + (t * n_heads + h_slot) * kHeadDim)
                    : (k + (t * n_kv_heads + (h_slot - n_heads)) * kHeadDim);

    if constexpr (kRotateHalf) {
      const float x0 = __half2float(vec[i]);
      const float x1 = __half2float(vec[i + kHalfD]);
      vec[i] = __float2half(x0 * c - x1 * sn);
      vec[i + kHalfD] = __float2half(x0 * sn + x1 * c);
    } else {
      const int i0 = 2 * i;
      const float x0 = __half2float(vec[i0]);
      const float x1 = __half2float(vec[i0 + 1]);
      vec[i0] = __float2half(x0 * c - x1 * sn);
      vec[i0 + 1] = __float2half(x0 * sn + x1 * c);
    }
  }
}

template <int kHeadDim>
void launch_rope(half* q, half* k, const float* cos_table, const float* sin_table,
                 std::int64_t s, std::int64_t n_heads, std::int64_t n_kv_heads,
                 std::int64_t pos_offset, RopeConvention convention,
                 cudaStream_t stream) {
  const std::int64_t total_pairs = s * (n_heads + n_kv_heads) * (kHeadDim / 2);
  const std::int64_t needed = (total_pairs + kRopeBlockSize - 1) / kRopeBlockSize;
  const int grid = static_cast<int>(std::min<std::int64_t>(needed, 65535));
  if (convention == RopeConvention::RotateHalf) {
    rope_kernel<kHeadDim, true><<<grid, kRopeBlockSize, 0, stream>>>(
        q, k, cos_table, sin_table, s, n_heads, n_kv_heads, pos_offset);
  } else {
    rope_kernel<kHeadDim, false><<<grid, kRopeBlockSize, 0, stream>>>(
        q, k, cos_table, sin_table, s, n_heads, n_kv_heads, pos_offset);
  }
  CUDA_CHECK_KERNEL();
}

}  // namespace

void rope(half* q, half* k, const float* cos_table, const float* sin_table,
          std::int64_t s, std::int64_t n_heads, std::int64_t n_kv_heads,
          std::int64_t head_dim, std::int64_t pos_offset, RopeConvention convention,
          cudaStream_t stream) {
  ENGINE_CHECK(q != nullptr, "rope: q is null");
  ENGINE_CHECK(n_kv_heads >= 0, "rope: n_kv_heads must be non-negative");
  if (n_kv_heads > 0) {
    ENGINE_CHECK(k != nullptr, "rope: k is null with n_kv_heads > 0");
  }
  ENGINE_CHECK(cos_table != nullptr, "rope: cos_table is null");
  ENGINE_CHECK(sin_table != nullptr, "rope: sin_table is null");
  ENGINE_CHECK(s >= 0, "rope: s must be non-negative");
  ENGINE_CHECK(n_heads > 0, "rope: n_heads must be positive");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128, "rope: head_dim must be 64 or 128");
  ENGINE_CHECK(pos_offset >= 0, "rope: pos_offset must be non-negative");
  ENGINE_CHECK(convention == RopeConvention::RotateHalf ||
                   convention == RopeConvention::AdjacentPair,
               "rope: unsupported RopeConvention");

  if (s == 0) return;

  if (head_dim == 64) {
    launch_rope<64>(q, k, cos_table, sin_table, s, n_heads, n_kv_heads, pos_offset,
                    convention, stream);
  } else {
    launch_rope<128>(q, k, cos_table, sin_table, s, n_heads, n_kv_heads, pos_offset,
                     convention, stream);
  }
}

}  // namespace engine::cuda
