//===----------------------------------------------------------------------===//
// kernels/attention_naive.cu -- Naive FP16 multi-head / GQA attention baseline.
//
// Materializes the full [n_heads, s_q, s_kv] FP32 score matrix in stream-
// ordered pool scratch memory, runs softmax_rows in-place, and multiplies
// probabilities by V in FP32 before storing FP16 output (ADR-0012).
//
// Exists intentionally as the O(N^2) memory & bandwidth baseline against
// which FlashAttention-2 (attention_prefill) is verified and benchmarked.
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

constexpr int kScoresWarps = 8;
constexpr int kLseBlockSize = 256;
constexpr int kPvTileJ = 256;

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_max(float* sdata, int tid, float val) {
  val = warp_reduce_max(val);
  if ((tid & 31) == 0) {
    sdata[tid >> 5] = val;
  }
  __syncthreads();
  if (tid < 32) {
    float v = (tid < (kLseBlockSize >> 5)) ? sdata[tid] : -INFINITY;
    v = warp_reduce_max(v);
    if (tid == 0) {
      sdata[0] = v;
    }
  }
  __syncthreads();
  return sdata[0];
}

__device__ __forceinline__ float block_reduce_sum(float* sdata, int tid, float val) {
  val = warp_reduce_sum(val);
  if ((tid & 31) == 0) {
    sdata[tid >> 5] = val;
  }
  __syncthreads();
  if (tid < 32) {
    float v = (tid < (kLseBlockSize >> 5)) ? sdata[tid] : 0.0f;
    v = warp_reduce_sum(v);
    if (tid == 0) {
      sdata[0] = v;
    }
  }
  __syncthreads();
  return sdata[0];
}

__global__ void fill_neg_inf_kernel(float* out, std::int64_t count) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t idx =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += stride) {
    out[idx] = -INFINITY;
  }
}

template <int kHeadDim>
__global__ void attention_naive_scores_kernel(
    const half* __restrict__ q, const half* __restrict__ k, float* __restrict__ scores,
    std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads, std::int64_t group_size,
    std::int64_t kv_head_stride, float scale, std::int64_t q_pos_offset, bool is_causal) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t h = row / s_q;
  const std::int64_t i = row % s_q;
  const std::int64_t h_kv = h / group_size;

  __shared__ float sq[kHeadDim];
  const half* q_row = q + ((i * n_heads) + h) * kHeadDim;
  for (int d = static_cast<int>(threadIdx.x); d < kHeadDim; d += blockDim.x) {
    sq[d] = __half2float(q_row[d]);
  }
  __syncthreads();

  const half* k_head = k + h_kv * kv_head_stride;
  float* row_scores = scores + row * s_kv;
  const int warp_id = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const std::int64_t total_warps = static_cast<std::int64_t>(gridDim.y) * kScoresWarps;
  const std::int64_t max_valid_j = q_pos_offset + i;

  for (std::int64_t j = static_cast<std::int64_t>(blockIdx.y) * kScoresWarps + warp_id;
       j < s_kv; j += total_warps) {
    if (is_causal && j > max_valid_j) {
      if (lane == 0) {
        row_scores[j] = -INFINITY;
      }
      continue;
    }

    const half* k_row = k_head + j * kHeadDim;
    float acc = 0.0f;
#pragma unroll
    for (int d = lane; d < kHeadDim; d += 32) {
      acc = fmaf(sq[d], __half2float(k_row[d]), acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      row_scores[j] = acc * scale;
    }
  }
}

__global__ void attention_naive_lse_kernel(const float* __restrict__ scores,
                                           float* __restrict__ out_lse, std::int64_t s_q,
                                           std::int64_t s_kv, std::int64_t n_heads) {
  __shared__ float sdata[kLseBlockSize >> 5];
  const int tid = static_cast<int>(threadIdx.x);
  const std::int64_t rows = n_heads * s_q;

  for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const std::int64_t h = r / s_q;
    const std::int64_t i = r % s_q;
    const float* row_scores = scores + r * s_kv;

    float thread_max = -INFINITY;
    for (std::int64_t c = tid; c < s_kv; c += kLseBlockSize) {
      thread_max = fmaxf(thread_max, row_scores[c]);
    }
    const float row_max = block_reduce_max(sdata, tid, thread_max);

    if (row_max == -INFINITY) {
      if (tid == 0) {
        out_lse[i * n_heads + h] = -INFINITY;
      }
      __syncthreads();
      continue;
    }

    float thread_sum = 0.0f;
    for (std::int64_t c = tid; c < s_kv; c += kLseBlockSize) {
      thread_sum += expf(row_scores[c] - row_max);
    }
    const float row_sum = block_reduce_sum(sdata, tid, thread_sum);
    if (tid == 0) {
      out_lse[i * n_heads + h] = row_max + logf(row_sum);
    }
    __syncthreads();
  }
}

template <int kHeadDim>
__global__ void attention_naive_pv_kernel(const float* __restrict__ probs,
                                          const half* __restrict__ v,
                                          half* __restrict__ out, std::int64_t s_q,
                                          std::int64_t s_kv, std::int64_t n_heads,
                                          std::int64_t group_size,
                                          std::int64_t kv_head_stride) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t h = row / s_q;
  const std::int64_t i = row % s_q;
  const std::int64_t h_kv = h / group_size;
  const int d = static_cast<int>(threadIdx.x);

  const float* row_probs = probs + row * s_kv;
  const half* v_head = v + h_kv * kv_head_stride;

  __shared__ float sp[kPvTileJ];
  float acc = 0.0f;

  for (std::int64_t j0 = 0; j0 < s_kv; j0 += kPvTileJ) {
    const std::int64_t rem = s_kv - j0;
    const int tile_len = static_cast<int>(rem < kPvTileJ ? rem : kPvTileJ);
    for (int t = d; t < tile_len; t += kHeadDim) {
      sp[t] = row_probs[j0 + t];
    }
    __syncthreads();

    for (int t = 0; t < tile_len; ++t) {
      acc = fmaf(sp[t], __half2float(v_head[(j0 + t) * kHeadDim + d]), acc);
    }
    __syncthreads();
  }

  out[((i * n_heads) + h) * kHeadDim + d] = __float2half(acc);
}

template <int kHeadDim>
void launch_attention_naive(const half* q, const half* k, const half* v, half* out,
                            std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                            std::int64_t group_size, std::int64_t kv_head_stride,
                            float scale, std::int64_t q_pos_offset, bool is_causal,
                            float* out_lse, cudaStream_t stream) {
  const std::int64_t rows = n_heads * s_q;
  const std::size_t score_bytes =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(s_kv) * sizeof(float);
  ScopedPoolBuffer scores_buf(pool_allocator(Device::CUDA), score_bytes, stream);
  auto* scores = static_cast<float*>(scores_buf.get());

  const std::int64_t scores_warps_needed = (s_kv + kScoresWarps - 1) / kScoresWarps;
  const std::int64_t max_j_blocks = (rows >= 2048) ? 8 : 64;
  const std::int64_t j_blocks = std::max<std::int64_t>(
      1, std::min<std::int64_t>(scores_warps_needed, max_j_blocks));
  const dim3 scores_grid(static_cast<unsigned>(rows), static_cast<unsigned>(j_blocks));
  attention_naive_scores_kernel<kHeadDim><<<scores_grid, kScoresWarps * 32, 0, stream>>>(
      q, k, scores, s_q, s_kv, n_heads, group_size, kv_head_stride, scale, q_pos_offset,
      is_causal);
  CUDA_CHECK_KERNEL();

  if (out_lse != nullptr) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    const std::int64_t max_blocks =
        std::max<std::int64_t>(1, static_cast<std::int64_t>(num_sms) * 16);
    const int lse_grid = static_cast<int>(std::min<std::int64_t>(rows, max_blocks));
    attention_naive_lse_kernel<<<lse_grid, kLseBlockSize, 0, stream>>>(
        scores, out_lse, s_q, s_kv, n_heads);
    CUDA_CHECK_KERNEL();
  }

  softmax_rows(scores, scores, rows, s_kv, stream);

  attention_naive_pv_kernel<kHeadDim>
      <<<static_cast<unsigned>(rows), kHeadDim, 0, stream>>>(
          scores, v, out, s_q, s_kv, n_heads, group_size, kv_head_stride);
  CUDA_CHECK_KERNEL();
}

}  // namespace

void attention_naive(const half* q, const half* k, const half* v, half* out,
                     std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                     std::int64_t n_kv_heads, std::int64_t head_dim,
                     std::int64_t kv_head_stride, float scale, std::int64_t q_pos_offset,
                     bool is_causal, float* out_lse, cudaStream_t stream) {
  ENGINE_CHECK(q != nullptr, "attention_naive: q is null");
  ENGINE_CHECK(k != nullptr, "attention_naive: k is null");
  ENGINE_CHECK(v != nullptr, "attention_naive: v is null");
  ENGINE_CHECK(out != nullptr, "attention_naive: out is null");
  ENGINE_CHECK(s_q >= 0, "attention_naive: s_q must be non-negative");
  ENGINE_CHECK(s_kv >= 0, "attention_naive: s_kv must be non-negative");
  ENGINE_CHECK(q_pos_offset >= 0, "attention_naive: q_pos_offset must be non-negative");
  ENGINE_CHECK(n_heads > 0, "attention_naive: n_heads must be positive");
  ENGINE_CHECK(n_kv_heads > 0, "attention_naive: n_kv_heads must be positive");
  ENGINE_CHECK(n_heads % n_kv_heads == 0,
               "attention_naive: n_heads must be divisible by n_kv_heads");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128,
               "attention_naive: head_dim must be 64 or 128");
  ENGINE_CHECK(std::isfinite(scale), "attention_naive: scale must be finite");
  const std::int64_t min_kv_stride = (s_kv > 0) ? (s_kv * head_dim) : 0;
  ENGINE_CHECK(kv_head_stride >= min_kv_stride,
               "attention_naive: kv_head_stride < s_kv * head_dim");

  if (s_q == 0) return;

  if (s_kv == 0) {
    const std::size_t out_bytes = static_cast<std::size_t>(s_q) *
                                  static_cast<std::size_t>(n_heads) *
                                  static_cast<std::size_t>(head_dim) * sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(out, 0, out_bytes, stream));
    if (out_lse != nullptr) {
      const std::int64_t lse_count = s_q * n_heads;
      const int grid = static_cast<int>((lse_count + 255) / 256);
      fill_neg_inf_kernel<<<grid, 256, 0, stream>>>(out_lse, lse_count);
      CUDA_CHECK_KERNEL();
    }
    return;
  }

  const std::int64_t group_size = n_heads / n_kv_heads;
  if (head_dim == 64) {
    launch_attention_naive<64>(q, k, v, out, s_q, s_kv, n_heads, group_size,
                               kv_head_stride, scale, q_pos_offset, is_causal, out_lse,
                               stream);
  } else {
    launch_attention_naive<128>(q, k, v, out, s_q, s_kv, n_heads, group_size,
                                kv_head_stride, scale, q_pos_offset, is_causal, out_lse,
                                stream);
  }
}

}  // namespace engine::cuda
