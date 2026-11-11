//===----------------------------------------------------------------------===//
// kernels/attention_prefill.cu -- FlashAttention-2 fused tiled prefill kernel.
//
// CUDA-core v1 implementation (ADR-0011, ADR-0012):
//   - Grid loops over query tiles (rows split across 8 warps = 256 threads)
//     and either attention heads (per-head mode) or KV heads (GQA group-
//     sharing mode for group_size in {2, 4, 8}).
//   - Query tile resides in shared memory; KV tiles stream through shared
//     memory with zero bank conflicts (half2 word stride + 2).
//   - Causal tile skipping omits all KV tiles strictly above the block's
//     diagonal and applies per-element causal masking only on boundary tiles.
//   - Online softmax (m, l, O) accumulates strictly in FP32 registers with a
//     single final normalization and rounding to FP16 on output store.
//   - O(S) auxiliary memory (no [H_q, s_q, s_kv] score matrix allocated).
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {
namespace {

constexpr int kPrefillWarps = 8;
constexpr int kPrefillBlockThreads = kPrefillWarps * 32;  // 256 threads

__global__ void fill_neg_inf_prefill_kernel(float* out, std::int64_t count) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += stride) {
    out[idx] = -INFINITY;
  }
}

template <int kHeadDim, int kBlockRows, int kBc, bool kGroupShared>
__global__ void flash_prefill_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ out, std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
    std::int64_t group_size, std::int64_t kv_head_stride, float scale,
    std::int64_t q_pos_offset, bool is_causal, float* __restrict__ out_lse) {
  static_assert(kHeadDim == 64 || kHeadDim == 128);
  static_assert(kBc == 32 || kBc == 64);
  static_assert(kBlockRows % (kPrefillWarps * 4) == 0);

  constexpr int kHalfDim = kHeadDim / 2;
  // Row stride in half2 (32-bit words): +2 avoids 32-bank conflicts across rows/keys.
  constexpr int kStrideHalf2 = kHalfDim + 2;
  constexpr int kMThr = kBlockRows / (kPrefillWarps * 4);
  constexpr int kKeysPerLane = kBc / 8;
  constexpr int kDimsPerLane = kHeadDim / 8;
  constexpr int kDimHalf2PerLane = kDimsPerLane / 2;

  extern __shared__ __align__(16) char smem_raw[];
  half2* q_smem_h2 = reinterpret_cast<half2*>(smem_raw);
  half2* k_smem_h2 = q_smem_h2 + kBlockRows * kStrideHalf2;
  half2* v_smem_h2 = k_smem_h2 + kBc * kStrideHalf2;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int sub_pair = lane >> 3;  // 0..3 within warp
  const int sub_col = lane & 7;    // 0..7 lanes sharing each row

  const int tokens_per_blk =
      kGroupShared ? (kBlockRows / static_cast<int>(group_size)) : kBlockRows;
  const std::int64_t q_tok_base = static_cast<std::int64_t>(blockIdx.x) * tokens_per_blk;
  const std::int64_t head_blk = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t kv_head = kGroupShared ? head_blk : (head_blk / group_size);

  // Cooperative load of Q tile [kBlockRows, kHeadDim] into shared memory.
  constexpr int kTotalQHalf2 = kBlockRows * kHalfDim;
  for (int idx = tid; idx < kTotalQHalf2; idx += kPrefillBlockThreads) {
    const int r = idx / kHalfDim;
    const int d2 = idx % kHalfDim;
    const std::int64_t q_tok = q_tok_base + (kGroupShared ? (r % tokens_per_blk) : r);
    const std::int64_t q_h =
        kGroupShared ? (kv_head * group_size + (r / tokens_per_blk)) : head_blk;
    half2 val = __float2half2_rn(0.0f);
    if (q_tok < s_q) {
      const half* q_ptr = q + ((q_tok * n_heads) + q_h) * kHeadDim + 2 * d2;
      val = *reinterpret_cast<const half2*>(q_ptr);
    }
    q_smem_h2[r * kStrideHalf2 + d2] = val;
  }

  // Row mapping owned by this thread: kMThr rows starting at r_base.
  const int r_base = warp_id * (4 * kMThr) + sub_pair * kMThr;
  std::int64_t row_q_tok[kMThr];
  std::int64_t row_q_head[kMThr];
  std::int64_t row_max_key[kMThr];
  float m_row[kMThr];
  float l_row[kMThr];
  float acc_o[kMThr][kDimsPerLane];

#pragma unroll
  for (int m = 0; m < kMThr; ++m) {
    const int r = r_base + m;
    const std::int64_t tok = q_tok_base + (kGroupShared ? (r % tokens_per_blk) : r);
    row_q_tok[m] = tok;
    row_q_head[m] =
        kGroupShared ? (kv_head * group_size + (r / tokens_per_blk)) : head_blk;
    if (tok >= s_q) {
      row_max_key[m] = -1;
    } else if (is_causal) {
      const std::int64_t q_glob = q_pos_offset + tok;
      row_max_key[m] = (q_glob < s_kv - 1) ? q_glob : (s_kv - 1);
    } else {
      row_max_key[m] = s_kv - 1;
    }
    m_row[m] = -INFINITY;
    l_row[m] = 0.0f;
#pragma unroll
    for (int di = 0; di < kDimsPerLane; ++di) {
      acc_o[m][di] = 0.0f;
    }
  }

  // Compute causal KV loop upper bound for the entire block (causal tile skipping).
  const std::int64_t rem_q = s_q - q_tok_base;
  const std::int64_t active_tokens = (rem_q < tokens_per_blk) ? rem_q : tokens_per_blk;
  const std::int64_t max_q_global = q_pos_offset + q_tok_base + active_tokens - 1;
  const std::int64_t kv_end =
      is_causal ? ((max_q_global + 1 < s_kv) ? (max_q_global + 1) : s_kv) : s_kv;
  const std::int64_t min_q_global = q_pos_offset + q_tok_base;

  const half* k_head_ptr = k + kv_head * kv_head_stride;
  const half* v_head_ptr = v + kv_head * kv_head_stride;
  constexpr int kTotalKVHalf2 = kBc * kHalfDim;

  for (std::int64_t j0 = 0; j0 < kv_end; j0 += kBc) {
    // Load K and V tiles cleanly into shared memory.
    for (int idx = tid; idx < kTotalKVHalf2; idx += kPrefillBlockThreads) {
      const int j_local = idx / kHalfDim;
      const int d2 = idx % kHalfDim;
      const std::int64_t j_glob = j0 + j_local;
      half2 kv_val = __float2half2_rn(0.0f);
      half2 vv_val = __float2half2_rn(0.0f);
      if (j_glob < s_kv) {
        kv_val = *reinterpret_cast<const half2*>(k_head_ptr + j_glob * kHeadDim + 2 * d2);
        vv_val = *reinterpret_cast<const half2*>(v_head_ptr + j_glob * kHeadDim + 2 * d2);
      }
      k_smem_h2[j_local * kStrideHalf2 + d2] = kv_val;
      v_smem_h2[j_local * kStrideHalf2 + d2] = vv_val;
    }
    __syncthreads();

    // 1. Tile score matrix S = Q_tile * K_tile^T in FP32 registers.
    float s[kMThr][kKeysPerLane];
#pragma unroll
    for (int m = 0; m < kMThr; ++m) {
#pragma unroll
      for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
        s[m][k_idx] = 0.0f;
      }
    }

#pragma unroll 4
    for (int d2 = 0; d2 < kHalfDim; ++d2) {
      float2 qv[kMThr];
#pragma unroll
      for (int m = 0; m < kMThr; ++m) {
        qv[m] = __half22float2(q_smem_h2[(r_base + m) * kStrideHalf2 + d2]);
      }
      float2 kv[kKeysPerLane];
#pragma unroll
      for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
        const int j_local = sub_col + k_idx * 8;
        kv[k_idx] = __half22float2(k_smem_h2[j_local * kStrideHalf2 + d2]);
      }
#pragma unroll
      for (int m = 0; m < kMThr; ++m) {
#pragma unroll
        for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
          s[m][k_idx] = fmaf(qv[m].x, kv[k_idx].x, s[m][k_idx]);
          s[m][k_idx] = fmaf(qv[m].y, kv[k_idx].y, s[m][k_idx]);
        }
      }
    }

    // 2. Scale, mask boundary/causal keys, and compute online softmax update.
    const bool interior_tile =
        (!is_causal && (j0 + kBc <= s_kv)) ||
        (is_causal && (j0 + kBc <= s_kv) && (j0 + kBc - 1 <= min_q_global));

#pragma unroll
    for (int m = 0; m < kMThr; ++m) {
      float row_tile_max = -INFINITY;
#pragma unroll
      for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
        const std::int64_t j_glob = j0 + sub_col + k_idx * 8;
        float val = s[m][k_idx] * scale;
        if (!interior_tile) {
          if (j_glob > row_max_key[m]) {
            val = -INFINITY;
          }
        } else if (row_max_key[m] < 0) {
          val = -INFINITY;
        }
        s[m][k_idx] = val;
        row_tile_max = fmaxf(row_tile_max, val);
      }

      row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffffu, row_tile_max, 4));
      row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffffu, row_tile_max, 2));
      row_tile_max = fmaxf(row_tile_max, __shfl_xor_sync(0xffffffffu, row_tile_max, 1));

      const bool tile_valid = (row_tile_max > -INFINITY);
      const float m_old = m_row[m];
      const float m_new = tile_valid ? fmaxf(m_old, row_tile_max) : m_old;
      const float alpha =
          tile_valid ? ((m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f) : 1.0f;

      float row_sum = 0.0f;
#pragma unroll
      for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
        const float p =
            (tile_valid && s[m][k_idx] > -INFINITY) ? expf(s[m][k_idx] - m_new) : 0.0f;
        s[m][k_idx] = p;  // overwrite S in place with P
        row_sum += p;
      }
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 4);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1);

      l_row[m] = l_row[m] * alpha + row_sum;
      m_row[m] = m_new;
#pragma unroll
      for (int di = 0; di < kDimsPerLane; ++di) {
        acc_o[m][di] *= alpha;
      }
    }

    // 3. Accumulate O += P * V using bank-conflict-free d2 = di2 * 8 + sub_col.
    const int src_base = sub_pair * 8;
    for (int src = 0; src < 8; ++src) {
      const int sender = src_base + src;
#pragma unroll
      for (int k_idx = 0; k_idx < kKeysPerLane; ++k_idx) {
        const int j_local = src + k_idx * 8;
        float2 vv[kDimHalf2PerLane];
#pragma unroll
        for (int di2 = 0; di2 < kDimHalf2PerLane; ++di2) {
          const int d2 = di2 * 8 + sub_col;
          vv[di2] = __half22float2(v_smem_h2[j_local * kStrideHalf2 + d2]);
        }
#pragma unroll
        for (int m = 0; m < kMThr; ++m) {
          const float p_val = __shfl_sync(0xffffffffu, s[m][k_idx], sender);
#pragma unroll
          for (int di2 = 0; di2 < kDimHalf2PerLane; ++di2) {
            acc_o[m][2 * di2] = fmaf(p_val, vv[di2].x, acc_o[m][2 * di2]);
            acc_o[m][2 * di2 + 1] = fmaf(p_val, vv[di2].y, acc_o[m][2 * di2 + 1]);
          }
        }
      }
    }

    __syncthreads();
  }

  // Final single normalization and writeback to token-major Out [s_q, n_heads, D].
#pragma unroll
  for (int m = 0; m < kMThr; ++m) {
    const std::int64_t tok = row_q_tok[m];
    if (tok < s_q) {
      const std::int64_t q_h = row_q_head[m];
      const bool has_valid = (m_row[m] > -INFINITY) && (l_row[m] > 0.0f);
      const float inv_l = has_valid ? (1.0f / l_row[m]) : 0.0f;

      half2* out_row_h2 =
          reinterpret_cast<half2*>(out + ((tok * n_heads) + q_h) * kHeadDim);
#pragma unroll
      for (int di2 = 0; di2 < kDimHalf2PerLane; ++di2) {
        const int d2 = di2 * 8 + sub_col;
        out_row_h2[d2] = has_valid ? __floats2half2_rn(acc_o[m][2 * di2] * inv_l,
                                                       acc_o[m][2 * di2 + 1] * inv_l)
                                   : __float2half2_rn(0.0f);
      }

      if (out_lse != nullptr && sub_col == 0) {
        out_lse[tok * n_heads + q_h] =
            has_valid ? (m_row[m] + logf(l_row[m])) : -INFINITY;
      }
    }
  }
}

template <int kHeadDim, int kBlockRows, int kBc, bool kGroupShared>
void dispatch_flash_kernel(const half* q, const half* k, const half* v, half* out,
                           std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                           std::int64_t n_kv_heads, std::int64_t group_size,
                           std::int64_t kv_head_stride, float scale,
                           std::int64_t q_pos_offset, bool is_causal, float* out_lse,
                           cudaStream_t stream) {
  constexpr int kStrideHalf2 = (kHeadDim / 2) + 2;
  constexpr std::size_t smem_bytes =
      static_cast<std::size_t>(kBlockRows + 2 * kBc) * kStrideHalf2 * sizeof(half2);

  if constexpr (smem_bytes > 49152) {
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_prefill_kernel<kHeadDim, kBlockRows, kBc, kGroupShared>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
  }

  const int tokens_per_blk =
      kGroupShared ? (kBlockRows / static_cast<int>(group_size)) : kBlockRows;
  const std::int64_t q_tiles = (s_q + tokens_per_blk - 1) / tokens_per_blk;
  const std::int64_t y_blocks = kGroupShared ? n_kv_heads : n_heads;
  const dim3 grid(static_cast<unsigned>(q_tiles), static_cast<unsigned>(y_blocks));

  flash_prefill_kernel<kHeadDim, kBlockRows, kBc, kGroupShared>
      <<<grid, kPrefillBlockThreads, smem_bytes, stream>>>(
          q, k, v, out, s_q, s_kv, n_heads, group_size, kv_head_stride, scale,
          q_pos_offset, is_causal, out_lse);
  CUDA_CHECK_KERNEL();
}

void run_attention_prefill(const half* q, const half* k, const half* v, half* out,
                           std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                           std::int64_t n_kv_heads, std::int64_t head_dim,
                           std::int64_t kv_head_stride, float scale,
                           std::int64_t q_pos_offset, bool is_causal, float* out_lse,
                           bool share_gqa_kv, cudaStream_t stream) {
  ENGINE_CHECK(q != nullptr, "attention_prefill: q is null");
  ENGINE_CHECK(k != nullptr, "attention_prefill: k is null");
  ENGINE_CHECK(v != nullptr, "attention_prefill: v is null");
  ENGINE_CHECK(out != nullptr, "attention_prefill: out is null");
  ENGINE_CHECK(s_q >= 0, "attention_prefill: s_q must be non-negative");
  ENGINE_CHECK(s_kv >= 0, "attention_prefill: s_kv must be non-negative");
  ENGINE_CHECK(q_pos_offset >= 0, "attention_prefill: q_pos_offset must be non-negative");
  ENGINE_CHECK(n_heads > 0, "attention_prefill: n_heads must be positive");
  ENGINE_CHECK(n_kv_heads > 0, "attention_prefill: n_kv_heads must be positive");
  ENGINE_CHECK(n_heads % n_kv_heads == 0,
               "attention_prefill: n_heads must be divisible by n_kv_heads");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128,
               "attention_prefill: head_dim must be 64 or 128");
  ENGINE_CHECK(std::isfinite(scale), "attention_prefill: scale must be finite");
  const std::int64_t min_kv_stride = (s_kv > 0) ? (s_kv * head_dim) : 0;
  ENGINE_CHECK(kv_head_stride >= min_kv_stride,
               "attention_prefill: kv_head_stride < s_kv * head_dim");

  if (s_q == 0) return;

  if (s_kv == 0) {
    const std::size_t out_bytes = static_cast<std::size_t>(s_q) *
                                  static_cast<std::size_t>(n_heads) *
                                  static_cast<std::size_t>(head_dim) * sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(out, 0, out_bytes, stream));
    if (out_lse != nullptr) {
      const std::int64_t lse_count = s_q * n_heads;
      const int grid = static_cast<int>((lse_count + 255) / 256);
      fill_neg_inf_prefill_kernel<<<grid, 256, 0, stream>>>(out_lse, lse_count);
      CUDA_CHECK_KERNEL();
    }
    return;
  }

  const std::int64_t group_size = n_heads / n_kv_heads;
  const bool use_group_shared =
      share_gqa_kv && (group_size == 2 || group_size == 4 || group_size == 8);

  if (head_dim == 64) {
    if (use_group_shared) {
      dispatch_flash_kernel<64, 128, 32, true>(
          q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, group_size, kv_head_stride, scale,
          q_pos_offset, is_causal, out_lse, stream);
    } else {
      dispatch_flash_kernel<64, 64, 64, false>(
          q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, group_size, kv_head_stride, scale,
          q_pos_offset, is_causal, out_lse, stream);
    }
  } else {
    if (use_group_shared) {
      dispatch_flash_kernel<128, 128, 32, true>(
          q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, group_size, kv_head_stride, scale,
          q_pos_offset, is_causal, out_lse, stream);
    } else {
      dispatch_flash_kernel<128, 64, 32, false>(
          q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, group_size, kv_head_stride, scale,
          q_pos_offset, is_causal, out_lse, stream);
    }
  }
}

}  // namespace

namespace detail {

void attention_prefill_with_mode(const half* q, const half* k, const half* v, half* out,
                                 std::int64_t s_q, std::int64_t s_kv,
                                 std::int64_t n_heads, std::int64_t n_kv_heads,
                                 std::int64_t head_dim, std::int64_t kv_head_stride,
                                 float scale, std::int64_t q_pos_offset, bool is_causal,
                                 float* out_lse, bool share_gqa_kv, cudaStream_t stream) {
  run_attention_prefill(q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim,
                        kv_head_stride, scale, q_pos_offset, is_causal, out_lse,
                        share_gqa_kv, stream);
}

}  // namespace detail

void attention_prefill(const half* q, const half* k, const half* v, half* out,
                       std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                       std::int64_t n_kv_heads, std::int64_t head_dim,
                       std::int64_t kv_head_stride, float scale,
                       std::int64_t q_pos_offset, bool is_causal, float* out_lse,
                       cudaStream_t stream) {
  run_attention_prefill(q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim,
                        kv_head_stride, scale, q_pos_offset, is_causal, out_lse,
                        /*share_gqa_kv=*/true, stream);
}

}  // namespace engine::cuda
