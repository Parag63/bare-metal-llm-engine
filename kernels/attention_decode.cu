//===----------------------------------------------------------------------===//
// kernels/attention_decode.cu -- Single-token decode attention (FlashDecoding).
//
// Splits KV sequence into chunks across thread blocks so all SMs saturate
// memory bandwidth during s_q == 1 decode even when n_kv_heads (e.g., 4 in
// TinyLlama) is far smaller than the GPU's 56 SMs (ADR-0013):
//   - Pass 1 (split kernel): Grid (n_kv_heads * splits) or (n_heads * splits).
//     Each thread loads 8 FP16 elements via 128-bit LDG.128 (uint4) and
//     reuses K/V across all Q heads in the GQA group held in shared memory.
//     Zero __syncthreads() inside the KV streaming loop; warp-uniform loop
//     bounds ensure convergent __shfl_xor_sync across all 32 lanes.
//   - Pass 2 (reduce kernel, skipped when splits == 1): Combines partial
//     (m, l, O) outputs from ScopedPoolBuffer across splits using exact LSE
//     rescaling before writing FP16 output.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>
#include <engine/pool_allocator.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {
namespace {

constexpr int kDecodeWarps = 8;
constexpr int kDecodeBlockThreads = kDecodeWarps * 32;  // 256

__device__ __forceinline__ float sub_warp_sum(float val, int lanes_per_key) {
  for (int offset = lanes_per_key >> 1; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

template <bool kVec16>
__device__ __forceinline__ void load_8_halfs(const half* ptr, half2 (&out_h2)[4]) {
  if constexpr (kVec16) {
    const uint4 v = *reinterpret_cast<const uint4*>(ptr);
    *reinterpret_cast<uint4*>(out_h2) = v;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out_h2[i] = __halves2half2(ptr[2 * i], ptr[2 * i + 1]);
    }
  }
}

template <int kHeadDim, int kGChunk, bool kVec16, bool kWritePartials>
__global__ void flash_decode_split_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ out, float* __restrict__ partial_o, float* __restrict__ partial_ml,
    std::int64_t kv_len, std::int64_t group_size, std::int64_t kv_head_stride,
    float scale, int num_splits, int q_heads_per_block) {
  static_assert(kHeadDim == 64 || kHeadDim == 128);
  static_assert(kGChunk == 1 || kGChunk == 2 || kGChunk == 4 || kGChunk == 8);

  constexpr int kDimsPerThread = 8;
  constexpr int kHalf2PerThread = kDimsPerThread / 2;      // 4
  constexpr int kLanesPerKey = kHeadDim / kDimsPerThread;  // 8 (D=64) or 16 (D=128)
  constexpr int kKeysPerWarp = 32 / kLanesPerKey;          // 4 (D=64) or 2 (D=128)

  const std::int64_t head_blk = static_cast<std::int64_t>(blockIdx.x);
  const int split_id = static_cast<int>(blockIdx.y);

  const std::int64_t q_head_start = head_blk * q_heads_per_block;
  const std::int64_t kv_head = q_head_start / group_size;

  const std::int64_t j_start =
      (kv_len * static_cast<std::int64_t>(split_id)) / num_splits;
  const std::int64_t j_end =
      (kv_len * static_cast<std::int64_t>(split_id + 1)) / num_splits;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int sub_key = lane / kLanesPerKey;    // 0..kKeysPerWarp-1
  const int col = lane & (kLanesPerKey - 1);  // 0..kLanesPerKey-1
  const int d_base = col * kDimsPerThread;

  __shared__ __align__(16) half2 s_q[kGChunk][kHeadDim / 2];
  __shared__ float s_o[kDecodeWarps][kHeadDim];
  __shared__ float s_m[kDecodeWarps];
  __shared__ float s_l[kDecodeWarps];

  const half* k_head_ptr = k + kv_head * kv_head_stride;
  const half* v_head_ptr = v + kv_head * kv_head_stride;

  for (int g_off = 0; g_off < q_heads_per_block; g_off += kGChunk) {
    // Cooperatively load kGChunk Q heads into shared memory.
    constexpr int kQHalf2Total = kGChunk * (kHeadDim / 2);
    for (int idx = tid; idx < kQHalf2Total; idx += kDecodeBlockThreads) {
      const int lg = idx / (kHeadDim / 2);
      const int d2 = idx % (kHeadDim / 2);
      const half* qp = q + (q_head_start + g_off + lg) * kHeadDim + 2 * d2;
      s_q[lg][d2] = __halves2half2(qp[0], qp[1]);
    }
    __syncthreads();

    float m_loc[kGChunk];
    float l_loc[kGChunk];
    float o_loc[kGChunk][kDimsPerThread];
#pragma unroll
    for (int lg = 0; lg < kGChunk; ++lg) {
      m_loc[lg] = -INFINITY;
      l_loc[lg] = 0.0f;
#pragma unroll
      for (int di = 0; di < kDimsPerThread; ++di) {
        o_loc[lg][di] = 0.0f;
      }
    }

    // Warp-uniform KV loop bound guarantees all 32 lanes converge for __shfl_xor_sync.
    const std::int64_t step = static_cast<std::int64_t>(kDecodeWarps) * kKeysPerWarp;
    for (std::int64_t j_warp_base =
             j_start + static_cast<std::int64_t>(warp_id) * kKeysPerWarp;
         j_warp_base < j_end; j_warp_base += step) {
      const std::int64_t j = j_warp_base + sub_key;
      const bool valid_key = (j < j_end);

      half2 k_h2[kHalf2PerThread];
      float v_f[kDimsPerThread];
      if (valid_key) {
        load_8_halfs<kVec16>(k_head_ptr + j * kHeadDim + d_base, k_h2);
        half2 v_h2[kHalf2PerThread];
        load_8_halfs<kVec16>(v_head_ptr + j * kHeadDim + d_base, v_h2);
#pragma unroll
        for (int di2 = 0; di2 < kHalf2PerThread; ++di2) {
          const float2 vf = __half22float2(v_h2[di2]);
          v_f[2 * di2] = vf.x;
          v_f[2 * di2 + 1] = vf.y;
        }
      }

#pragma unroll
      for (int lg = 0; lg < kGChunk; ++lg) {
        float partial = 0.0f;
        if (valid_key) {
#pragma unroll
          for (int di2 = 0; di2 < kHalf2PerThread; ++di2) {
            const float2 kf = __half22float2(k_h2[di2]);
            const float2 qf = __half22float2(s_q[lg][col * kHalf2PerThread + di2]);
            partial = fmaf(qf.x, kf.x, partial);
            partial = fmaf(qf.y, kf.y, partial);
          }
        }
        const float score = sub_warp_sum(partial, kLanesPerKey) * scale;

        if (valid_key) {
          const float m_old = m_loc[lg];
          const float m_new = fmaxf(m_old, score);
          const float alpha = (m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f;
          const float p = expf(score - m_new);

          l_loc[lg] = l_loc[lg] * alpha + p;
          m_loc[lg] = m_new;
#pragma unroll
          for (int di = 0; di < kDimsPerThread; ++di) {
            o_loc[lg][di] = fmaf(p, v_f[di], o_loc[lg][di] * alpha);
          }
        }
      }
    }

    // Merge kKeysPerWarp independent sub-key lanes inside each warp.
#pragma unroll
    for (int lg = 0; lg < kGChunk; ++lg) {
      for (int offset = kKeysPerWarp >> 1; offset > 0; offset >>= 1) {
        const int shift = offset * kLanesPerKey;
        const float m_other = __shfl_xor_sync(0xffffffffu, m_loc[lg], shift);
        const float l_other = __shfl_xor_sync(0xffffffffu, l_loc[lg], shift);
        const float m_new = fmaxf(m_loc[lg], m_other);
        const float scale_a =
            (m_loc[lg] > -INFINITY)
                ? ((m_loc[lg] == m_new) ? 1.0f : expf(m_loc[lg] - m_new))
                : 0.0f;
        const float scale_b = (m_other > -INFINITY)
                                  ? ((m_other == m_new) ? 1.0f : expf(m_other - m_new))
                                  : 0.0f;
        l_loc[lg] = l_loc[lg] * scale_a + l_other * scale_b;
        m_loc[lg] = m_new;
#pragma unroll
        for (int di = 0; di < kDimsPerThread; ++di) {
          const float o_other = __shfl_xor_sync(0xffffffffu, o_loc[lg][di], shift);
          o_loc[lg][di] = o_loc[lg][di] * scale_a + o_other * scale_b;
        }
      }
    }

    // Combine across warps via shared memory and emit output or partial state.
#pragma unroll
    for (int lg = 0; lg < kGChunk; ++lg) {
      if (sub_key == 0) {
        if (col == 0) {
          s_m[warp_id] = m_loc[lg];
          s_l[warp_id] = l_loc[lg];
        }
#pragma unroll
        for (int di = 0; di < kDimsPerThread; ++di) {
          s_o[warp_id][d_base + di] = o_loc[lg][di];
        }
      }
      __syncthreads();

      if (tid < kHeadDim) {
        float m_blk = -INFINITY;
#pragma unroll
        for (int w = 0; w < kDecodeWarps; ++w) {
          m_blk = fmaxf(m_blk, s_m[w]);
        }

        const std::int64_t h_glob = q_head_start + g_off + lg;
        if (m_blk == -INFINITY) {
          if constexpr (kWritePartials) {
            const std::int64_t slot = h_glob * num_splits + split_id;
            partial_o[slot * kHeadDim + tid] = 0.0f;
            if (tid == 0) {
              partial_ml[slot * 2 + 0] = -INFINITY;
              partial_ml[slot * 2 + 1] = 0.0f;
            }
          } else {
            out[h_glob * kHeadDim + tid] = __float2half(0.0f);
          }
        } else {
          float l_blk = 0.0f;
          float o_blk = 0.0f;
#pragma unroll
          for (int w = 0; w < kDecodeWarps; ++w) {
            const float mw = s_m[w];
            const float lw = s_l[w];
            if (mw > -INFINITY && lw > 0.0f) {
              const float scale_w = (mw == m_blk) ? 1.0f : expf(mw - m_blk);
              l_blk += lw * scale_w;
              o_blk = fmaf(s_o[w][tid], scale_w, o_blk);
            }
          }

          if constexpr (kWritePartials) {
            const std::int64_t slot = h_glob * num_splits + split_id;
            partial_o[slot * kHeadDim + tid] = o_blk;
            if (tid == 0) {
              partial_ml[slot * 2 + 0] = m_blk;
              partial_ml[slot * 2 + 1] = l_blk;
            }
          } else {
            const float res = (l_blk > 0.0f) ? (o_blk / l_blk) : 0.0f;
            out[h_glob * kHeadDim + tid] = __float2half(res);
          }
        }
      }
      __syncthreads();
    }
  }
}

template <int kHeadDim, int kGroupSize, bool kVec16, bool kWritePartials>
__global__ void flash_decode_gqa_warp_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ out, float* __restrict__ partial_o, float* __restrict__ partial_ml,
    std::int64_t kv_len, std::int64_t kv_head_stride, float scale, int num_splits) {
  static_assert(kHeadDim == 64 || kHeadDim == 128);
  static_assert(kGroupSize == 2 || kGroupSize == 4 || kGroupSize == 8);

  constexpr int kNumWarps = kGroupSize;
  constexpr int kBlockThreads = kNumWarps * 32;
  constexpr int kDimsPerThread = 8;
  constexpr int kHalf2PerThread = kDimsPerThread / 2;      // 4
  constexpr int kLanesPerKey = kHeadDim / kDimsPerThread;  // 8 (D=64) or 16 (D=128)
  constexpr int kKeysPerWarp = 32 / kLanesPerKey;          // 4 (D=64) or 2 (D=128)
  constexpr int kTileK = kBlockThreads / kLanesPerKey;

  const std::int64_t kv_head = static_cast<std::int64_t>(blockIdx.x);
  const int split_id = static_cast<int>(blockIdx.y);

  const std::int64_t j_start =
      (kv_len * static_cast<std::int64_t>(split_id)) / num_splits;
  const std::int64_t j_end =
      (kv_len * static_cast<std::int64_t>(split_id + 1)) / num_splits;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int sub_key = lane / kLanesPerKey;
  const int col = lane & (kLanesPerKey - 1);
  const int d_base = col * kDimsPerThread;

  const int load_k = tid / kLanesPerKey;
  const int load_col = tid & (kLanesPerKey - 1);
  const int load_d_base = load_col * kDimsPerThread;

  __shared__ __align__(16) half2 s_k[kTileK][kHeadDim / 2];
  __shared__ __align__(16) half2 s_v[kTileK][kHeadDim / 2];

  // Each warp owns a single Q head in the GQA group and holds it in registers.
  const std::int64_t h_glob = kv_head * kGroupSize + warp_id;
  half2 q_h2[kHalf2PerThread];
  load_8_halfs<false>(q + h_glob * kHeadDim + d_base, q_h2);
  float q_f[kDimsPerThread];
#pragma unroll
  for (int di2 = 0; di2 < kHalf2PerThread; ++di2) {
    const float2 qf = __half22float2(q_h2[di2]);
    q_f[2 * di2] = qf.x;
    q_f[2 * di2 + 1] = qf.y;
  }

  float m_loc = -INFINITY;
  float l_loc = 0.0f;
  float o_loc[kDimsPerThread];
#pragma unroll
  for (int di = 0; di < kDimsPerThread; ++di) {
    o_loc[di] = 0.0f;
  }

  const half* k_head_ptr = k + kv_head * kv_head_stride;
  const half* v_head_ptr = v + kv_head * kv_head_stride;

  for (std::int64_t tile_start = j_start; tile_start < j_end; tile_start += kTileK) {
    const std::int64_t load_j = tile_start + load_k;
    if (load_j < j_end) {
      half2 k_reg[kHalf2PerThread];
      half2 v_reg[kHalf2PerThread];
      load_8_halfs<kVec16>(k_head_ptr + load_j * kHeadDim + load_d_base, k_reg);
      load_8_halfs<kVec16>(v_head_ptr + load_j * kHeadDim + load_d_base, v_reg);
      *reinterpret_cast<uint4*>(&s_k[load_k][load_col * kHalf2PerThread]) =
          *reinterpret_cast<const uint4*>(k_reg);
      *reinterpret_cast<uint4*>(&s_v[load_k][load_col * kHalf2PerThread]) =
          *reinterpret_cast<const uint4*>(v_reg);
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < kTileK; step += kKeysPerWarp) {
      const int k_idx = step + sub_key;
      const bool valid_key = (tile_start + k_idx) < j_end;

      float partial = 0.0f;
      if (valid_key) {
        const uint4 k_u4 =
            *reinterpret_cast<const uint4*>(&s_k[k_idx][col * kHalf2PerThread]);
        half2 k_h2[kHalf2PerThread];
        *reinterpret_cast<uint4*>(k_h2) = k_u4;
#pragma unroll
        for (int di2 = 0; di2 < kHalf2PerThread; ++di2) {
          const float2 kf = __half22float2(k_h2[di2]);
          partial = fmaf(q_f[2 * di2], kf.x, partial);
          partial = fmaf(q_f[2 * di2 + 1], kf.y, partial);
        }
      }
      const float score = sub_warp_sum(partial, kLanesPerKey) * scale;

      if (valid_key) {
        const uint4 v_u4 =
            *reinterpret_cast<const uint4*>(&s_v[k_idx][col * kHalf2PerThread]);
        half2 v_h2[kHalf2PerThread];
        *reinterpret_cast<uint4*>(v_h2) = v_u4;

        const float m_old = m_loc;
        const float m_new = fmaxf(m_old, score);
        const float alpha = (m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f;
        const float p = expf(score - m_new);

        l_loc = l_loc * alpha + p;
        m_loc = m_new;
#pragma unroll
        for (int di2 = 0; di2 < kHalf2PerThread; ++di2) {
          const float2 vf = __half22float2(v_h2[di2]);
          o_loc[2 * di2] = fmaf(p, vf.x, o_loc[2 * di2] * alpha);
          o_loc[2 * di2 + 1] = fmaf(p, vf.y, o_loc[2 * di2 + 1] * alpha);
        }
      }
    }
    __syncthreads();
  }

  // Merge kKeysPerWarp independent sub-keys inside each warp (zero cross-warp sync!).
  for (int offset = kKeysPerWarp >> 1; offset > 0; offset >>= 1) {
    const int shift = offset * kLanesPerKey;
    const float m_other = __shfl_xor_sync(0xffffffffu, m_loc, shift);
    const float l_other = __shfl_xor_sync(0xffffffffu, l_loc, shift);
    const float m_new = fmaxf(m_loc, m_other);
    const float scale_a =
        (m_loc > -INFINITY) ? ((m_loc == m_new) ? 1.0f : expf(m_loc - m_new)) : 0.0f;
    const float scale_b = (m_other > -INFINITY)
                              ? ((m_other == m_new) ? 1.0f : expf(m_other - m_new))
                              : 0.0f;
    l_loc = l_loc * scale_a + l_other * scale_b;
    m_loc = m_new;
#pragma unroll
    for (int di = 0; di < kDimsPerThread; ++di) {
      const float o_other = __shfl_xor_sync(0xffffffffu, o_loc[di], shift);
      o_loc[di] = o_loc[di] * scale_a + o_other * scale_b;
    }
  }

  if (sub_key == 0) {
    if constexpr (kWritePartials) {
      const std::int64_t slot = h_glob * num_splits + split_id;
      if (m_loc == -INFINITY) {
#pragma unroll
        for (int di = 0; di < kDimsPerThread; ++di) {
          partial_o[slot * kHeadDim + d_base + di] = 0.0f;
        }
        if (col == 0) {
          partial_ml[slot * 2 + 0] = -INFINITY;
          partial_ml[slot * 2 + 1] = 0.0f;
        }
      } else {
        *reinterpret_cast<float4*>(&partial_o[slot * kHeadDim + d_base + 0]) =
            make_float4(o_loc[0], o_loc[1], o_loc[2], o_loc[3]);
        *reinterpret_cast<float4*>(&partial_o[slot * kHeadDim + d_base + 4]) =
            make_float4(o_loc[4], o_loc[5], o_loc[6], o_loc[7]);
        if (col == 0) {
          partial_ml[slot * 2 + 0] = m_loc;
          partial_ml[slot * 2 + 1] = l_loc;
        }
      }
    } else {
      if (m_loc == -INFINITY || l_loc <= 0.0f) {
#pragma unroll
        for (int di = 0; di < kDimsPerThread; ++di) {
          out[h_glob * kHeadDim + d_base + di] = __float2half(0.0f);
        }
      } else {
        const float inv_l = 1.0f / l_loc;
#pragma unroll
        for (int di = 0; di < kDimsPerThread; ++di) {
          out[h_glob * kHeadDim + d_base + di] = __float2half(o_loc[di] * inv_l);
        }
      }
    }
  }
}

template <int kHeadDim>
__global__ void flash_decode_reduce_kernel(const float* __restrict__ partial_o,
                                           const float* __restrict__ partial_ml,
                                           half* __restrict__ out, int num_splits) {
  const std::int64_t h = static_cast<std::int64_t>(blockIdx.x);
  const int d = static_cast<int>(threadIdx.x);
  const std::int64_t base_slot = h * num_splits;

  float m_glob = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    m_glob = fmaxf(m_glob, partial_ml[(base_slot + s) * 2 + 0]);
  }

  if (m_glob == -INFINITY) {
    out[h * kHeadDim + d] = __float2half(0.0f);
    return;
  }

  float l_glob = 0.0f;
  float o_glob = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float ms = partial_ml[(base_slot + s) * 2 + 0];
    const float ls = partial_ml[(base_slot + s) * 2 + 1];
    if (ms > -INFINITY && ls > 0.0f) {
      const float w = (ms == m_glob) ? 1.0f : expf(ms - m_glob);
      l_glob += ls * w;
      o_glob = fmaf(partial_o[(base_slot + s) * kHeadDim + d], w, o_glob);
    }
  }

  const float val = (l_glob > 0.0f) ? (o_glob / l_glob) : 0.0f;
  out[h * kHeadDim + d] = __float2half(val);
}

int choose_auto_splits(std::int64_t kv_len, std::int64_t y_blocks) {
  // Calibrated from the FlashDecoding split-KV sweep on sm_89 (56 SMs):
  //   - Target 2 * num_sms total blocks (2 waves / 2 blocks per SM) to hide
  //     DRAM latency while bounding partial_o + reduce overhead.
  //   - Enforce a minimum keys/split threshold (128 when GQA-limited, 512 when
  //     y_blocks already fills >= half the GPU) so short sequences do not
  //     spend more time launching/reducing partials than reading KV.
  //   - Snap splits >= per_wave down to an exact SM-wave multiple when y_blocks
  //     divides num_sms so no small tail wave stalls completion.
  if (kv_len < 256) {
    return 1;
  }
  int device = 0;
  int num_sms = 56;
  if (cudaGetDevice(&device) == cudaSuccess) {
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  }
  const int target_blocks = std::max(1, num_sms * 2);
  const int wanted = static_cast<int>(
      (static_cast<std::int64_t>(target_blocks) + y_blocks - 1) / y_blocks);
  const std::int64_t min_keys_per_split = (y_blocks * 2 >= num_sms) ? 512 : 128;
  const int max_by_len = static_cast<int>(kv_len / min_keys_per_split);
  int splits = std::min({wanted, max_by_len, 64});
  if (y_blocks > 0 && (num_sms % y_blocks) == 0) {
    const int per_wave = static_cast<int>(num_sms / y_blocks);
    if (per_wave > 1 && splits > per_wave) {
      splits = (splits / per_wave) * per_wave;
    }
  }
  return std::max(1, splits);
}

template <int kHeadDim, int kGChunk, bool kVec16>
void dispatch_decode(const half* q, const half* k, const half* v, half* out,
                     std::int64_t kv_len, std::int64_t n_heads, std::int64_t group_size,
                     std::int64_t kv_head_stride, float scale, int split_kv,
                     int q_heads_per_block, cudaStream_t stream) {
  const std::int64_t head_blocks = n_heads / q_heads_per_block;
  const int req_splits =
      (split_kv == 0) ? choose_auto_splits(kv_len, head_blocks) : split_kv;
  const int actual_splits =
      std::max(1, std::min<int>(req_splits, static_cast<int>(kv_len)));

  const dim3 grid(static_cast<unsigned>(head_blocks),
                  static_cast<unsigned>(actual_splits));

  if (actual_splits == 1) {
    flash_decode_split_kernel<kHeadDim, kGChunk, kVec16, false>
        <<<grid, kDecodeBlockThreads, 0, stream>>>(q, k, v, out, nullptr, nullptr, kv_len,
                                                   group_size, kv_head_stride, scale, 1,
                                                   q_heads_per_block);
    CUDA_CHECK_KERNEL();
    return;
  }

  const std::size_t o_elems = static_cast<std::size_t>(n_heads) *
                              static_cast<std::size_t>(actual_splits) * kHeadDim;
  const std::size_t ml_elems =
      static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(actual_splits) * 2;
  const std::size_t o_bytes = o_elems * sizeof(float);
  const std::size_t total_bytes = o_bytes + ml_elems * sizeof(float);

  ScopedPoolBuffer scratch(pool_allocator(Device::CUDA), total_bytes, stream);
  auto* raw = static_cast<char*>(scratch.get());
  auto* partial_o = reinterpret_cast<float*>(raw);
  auto* partial_ml = reinterpret_cast<float*>(raw + o_bytes);

  flash_decode_split_kernel<kHeadDim, kGChunk, kVec16, true>
      <<<grid, kDecodeBlockThreads, 0, stream>>>(q, k, v, out, partial_o, partial_ml,
                                                 kv_len, group_size, kv_head_stride,
                                                 scale, actual_splits, q_heads_per_block);
  CUDA_CHECK_KERNEL();

  flash_decode_reduce_kernel<kHeadDim>
      <<<static_cast<unsigned>(n_heads), kHeadDim, 0, stream>>>(partial_o, partial_ml,
                                                                out, actual_splits);
  CUDA_CHECK_KERNEL();
}

template <int kHeadDim, int kGroupSize, bool kVec16>
void dispatch_gqa_warp(const half* q, const half* k, const half* v, half* out,
                       std::int64_t kv_len, std::int64_t n_heads,
                       std::int64_t kv_head_stride, float scale, int split_kv,
                       cudaStream_t stream) {
  constexpr int kBlockThreads = kGroupSize * 32;
  const std::int64_t head_blocks = n_heads / kGroupSize;
  const int req_splits =
      (split_kv == 0) ? choose_auto_splits(kv_len, head_blocks) : split_kv;
  const int actual_splits =
      std::max(1, std::min<int>(req_splits, static_cast<int>(kv_len)));

  const dim3 grid(static_cast<unsigned>(head_blocks),
                  static_cast<unsigned>(actual_splits));

  if (actual_splits == 1) {
    flash_decode_gqa_warp_kernel<kHeadDim, kGroupSize, kVec16, false>
        <<<grid, kBlockThreads, 0, stream>>>(q, k, v, out, nullptr, nullptr, kv_len,
                                             kv_head_stride, scale, 1);
    CUDA_CHECK_KERNEL();
    return;
  }

  const std::size_t o_elems = static_cast<std::size_t>(n_heads) *
                              static_cast<std::size_t>(actual_splits) * kHeadDim;
  const std::size_t ml_elems =
      static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(actual_splits) * 2;
  const std::size_t o_bytes = o_elems * sizeof(float);
  const std::size_t total_bytes = o_bytes + ml_elems * sizeof(float);

  ScopedPoolBuffer scratch(pool_allocator(Device::CUDA), total_bytes, stream);
  auto* raw = static_cast<char*>(scratch.get());
  auto* partial_o = reinterpret_cast<float*>(raw);
  auto* partial_ml = reinterpret_cast<float*>(raw + o_bytes);

  flash_decode_gqa_warp_kernel<kHeadDim, kGroupSize, kVec16, true>
      <<<grid, kBlockThreads, 0, stream>>>(q, k, v, out, partial_o, partial_ml, kv_len,
                                           kv_head_stride, scale, actual_splits);
  CUDA_CHECK_KERNEL();

  flash_decode_reduce_kernel<kHeadDim>
      <<<static_cast<unsigned>(n_heads), kHeadDim, 0, stream>>>(partial_o, partial_ml,
                                                                out, actual_splits);
  CUDA_CHECK_KERNEL();
}

template <int kHeadDim, bool kVec16>
void dispatch_decode_gchunk(const half* q, const half* k, const half* v, half* out,
                            std::int64_t kv_len, std::int64_t n_heads,
                            std::int64_t group_size, std::int64_t kv_head_stride,
                            float scale, int split_kv, int q_heads_per_block,
                            cudaStream_t stream) {
  if (q_heads_per_block == 8 && group_size == 8) {
    dispatch_gqa_warp<kHeadDim, 8, kVec16>(q, k, v, out, kv_len, n_heads, kv_head_stride,
                                           scale, split_kv, stream);
  } else if (q_heads_per_block == 4 && group_size == 4) {
    dispatch_gqa_warp<kHeadDim, 4, kVec16>(q, k, v, out, kv_len, n_heads, kv_head_stride,
                                           scale, split_kv, stream);
  } else if (q_heads_per_block == 2 && group_size == 2) {
    dispatch_gqa_warp<kHeadDim, 2, kVec16>(q, k, v, out, kv_len, n_heads, kv_head_stride,
                                           scale, split_kv, stream);
  } else if (q_heads_per_block == 8) {
    dispatch_decode<kHeadDim, 8, kVec16>(q, k, v, out, kv_len, n_heads, group_size,
                                         kv_head_stride, scale, split_kv,
                                         q_heads_per_block, stream);
  } else if (q_heads_per_block == 4) {
    dispatch_decode<kHeadDim, 4, kVec16>(q, k, v, out, kv_len, n_heads, group_size,
                                         kv_head_stride, scale, split_kv,
                                         q_heads_per_block, stream);
  } else if (q_heads_per_block == 2) {
    dispatch_decode<kHeadDim, 2, kVec16>(q, k, v, out, kv_len, n_heads, group_size,
                                         kv_head_stride, scale, split_kv,
                                         q_heads_per_block, stream);
  } else {
    dispatch_decode<kHeadDim, 1, kVec16>(q, k, v, out, kv_len, n_heads, group_size,
                                         kv_head_stride, scale, split_kv,
                                         q_heads_per_block, stream);
  }
}

void run_attention_decode(const half* q, const half* k, const half* v, half* out,
                          std::int64_t kv_len, std::int64_t n_heads,
                          std::int64_t n_kv_heads, std::int64_t head_dim,
                          std::int64_t kv_head_stride, float scale, int split_kv,
                          bool share_gqa_kv, cudaStream_t stream) {
  ENGINE_CHECK(q != nullptr, "attention_decode: q is null");
  ENGINE_CHECK(k != nullptr, "attention_decode: k is null");
  ENGINE_CHECK(v != nullptr, "attention_decode: v is null");
  ENGINE_CHECK(out != nullptr, "attention_decode: out is null");
  ENGINE_CHECK(kv_len >= 0, "attention_decode: kv_len must be non-negative");
  ENGINE_CHECK(n_heads > 0, "attention_decode: n_heads must be positive");
  ENGINE_CHECK(n_kv_heads > 0, "attention_decode: n_kv_heads must be positive");
  ENGINE_CHECK(n_heads % n_kv_heads == 0,
               "attention_decode: n_heads must be divisible by n_kv_heads");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128,
               "attention_decode: head_dim must be 64 or 128");
  ENGINE_CHECK(std::isfinite(scale), "attention_decode: scale must be finite");
  ENGINE_CHECK(split_kv >= 0, "attention_decode: split_kv must be non-negative");
  const std::int64_t min_kv_stride = (kv_len > 0) ? (kv_len * head_dim) : 0;
  ENGINE_CHECK(kv_head_stride >= min_kv_stride,
               "attention_decode: kv_head_stride < kv_len * head_dim");

  if (kv_len == 0) {
    const std::size_t out_bytes = static_cast<std::size_t>(n_heads) *
                                  static_cast<std::size_t>(head_dim) * sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(out, 0, out_bytes, stream));
    return;
  }

  const std::int64_t group_size = n_heads / n_kv_heads;
  const int q_heads_per_block =
      (share_gqa_kv && group_size > 1) ? static_cast<int>(group_size) : 1;

  const bool vec16 = ((reinterpret_cast<std::uintptr_t>(k) & 15u) == 0) &&
                     ((reinterpret_cast<std::uintptr_t>(v) & 15u) == 0) &&
                     ((kv_head_stride & 7) == 0);

  if (head_dim == 64) {
    if (vec16) {
      dispatch_decode_gchunk<64, true>(q, k, v, out, kv_len, n_heads, group_size,
                                       kv_head_stride, scale, split_kv, q_heads_per_block,
                                       stream);
    } else {
      dispatch_decode_gchunk<64, false>(q, k, v, out, kv_len, n_heads, group_size,
                                        kv_head_stride, scale, split_kv,
                                        q_heads_per_block, stream);
    }
  } else {
    if (vec16) {
      dispatch_decode_gchunk<128, true>(q, k, v, out, kv_len, n_heads, group_size,
                                        kv_head_stride, scale, split_kv,
                                        q_heads_per_block, stream);
    } else {
      dispatch_decode_gchunk<128, false>(q, k, v, out, kv_len, n_heads, group_size,
                                         kv_head_stride, scale, split_kv,
                                         q_heads_per_block, stream);
    }
  }
}

}  // namespace

namespace detail {

void attention_decode_with_mode(const half* q, const half* k, const half* v, half* out,
                                std::int64_t kv_len, std::int64_t n_heads,
                                std::int64_t n_kv_heads, std::int64_t head_dim,
                                std::int64_t kv_head_stride, float scale, int split_kv,
                                bool share_gqa_kv, cudaStream_t stream) {
  run_attention_decode(q, k, v, out, kv_len, n_heads, n_kv_heads, head_dim,
                       kv_head_stride, scale, split_kv, share_gqa_kv, stream);
}

}  // namespace detail

void attention_decode(const half* q, const half* k, const half* v, half* out,
                      std::int64_t kv_len, std::int64_t n_heads, std::int64_t n_kv_heads,
                      std::int64_t head_dim, std::int64_t kv_head_stride, float scale,
                      int split_kv, cudaStream_t stream) {
  run_attention_decode(q, k, v, out, kv_len, n_heads, n_kv_heads, head_dim,
                       kv_head_stride, scale, split_kv, /*share_gqa_kv=*/true, stream);
}

}  // namespace engine::cuda
