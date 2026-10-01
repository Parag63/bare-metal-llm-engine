//===----------------------------------------------------------------------===//
// kernels/embedding.cu -- Token embedding lookup for LLM inference.
//
// table: [vocab_size, hidden_dim], row-major
// input_ids: [num_tokens]
// out: [num_tokens, hidden_dim]
//
// For each token t in [0, num_tokens):
//   id = input_ids[t]
//   gather row id from table into row t of out.
//
// OPTIMIZATION:
// Uses 128-bit vector memory instructions (float4 for FP32, uint4 for FP16)
// to maximize DRAM bus utilization during token embedding gather.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

// Vectorized FP32 kernel: 4 floats (128 bits) per load/store
__global__ void embedding_f32_vec4(const float* __restrict__ table,
                                   const int32_t* __restrict__ input_ids,
                                   float* __restrict__ out, int64_t num_vecs,
                                   int64_t hidden_dim, int64_t vocab_size) {
  const int64_t t = blockIdx.y;
  const int64_t v_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v_idx >= num_vecs) return;

  const int32_t id = input_ids[t];
  float4* out_row = reinterpret_cast<float4*>(out + t * hidden_dim);

  if (id >= 0 && id < vocab_size) {
    const float4* in_row =
        reinterpret_cast<const float4*>(table + static_cast<int64_t>(id) * hidden_dim);
    out_row[v_idx] = in_row[v_idx];
  } else {
    out_row[v_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// Fallback scalar FP32 kernel
__global__ void embedding_f32_scalar(const float* __restrict__ table,
                                     const int32_t* __restrict__ input_ids,
                                     float* __restrict__ out, int64_t hidden_dim,
                                     int64_t vocab_size) {
  const int64_t t = blockIdx.y;
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= hidden_dim) return;

  const int32_t id = input_ids[t];
  if (id >= 0 && id < vocab_size) {
    out[t * hidden_dim + c] = table[static_cast<int64_t>(id) * hidden_dim + c];
  } else {
    out[t * hidden_dim + c] = 0.0f;
  }
}

// Vectorized FP16 kernel: 8 halfs (128 bits = uint4) per load/store
__global__ void embedding_fp16_vec8(const half* __restrict__ table,
                                    const int32_t* __restrict__ input_ids,
                                    half* __restrict__ out, int64_t num_vecs,
                                    int64_t hidden_dim, int64_t vocab_size) {
  const int64_t t = blockIdx.y;
  const int64_t v_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v_idx >= num_vecs) return;

  const int32_t id = input_ids[t];
  uint4* out_row = reinterpret_cast<uint4*>(out + t * hidden_dim);

  if (id >= 0 && id < vocab_size) {
    const uint4* in_row =
        reinterpret_cast<const uint4*>(table + static_cast<int64_t>(id) * hidden_dim);
    out_row[v_idx] = in_row[v_idx];
  } else {
    out_row[v_idx] = make_uint4(0, 0, 0, 0);
  }
}

// Fallback scalar FP16 kernel
__global__ void embedding_fp16_scalar(const half* __restrict__ table,
                                      const int32_t* __restrict__ input_ids,
                                      half* __restrict__ out, int64_t hidden_dim,
                                      int64_t vocab_size) {
  const int64_t t = blockIdx.y;
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= hidden_dim) return;

  const int32_t id = input_ids[t];
  if (id >= 0 && id < vocab_size) {
    out[t * hidden_dim + c] = table[static_cast<int64_t>(id) * hidden_dim + c];
  } else {
    out[t * hidden_dim + c] = __float2half(0.0f);
  }
}

}  // namespace

void embedding(const float* table, const std::int32_t* input_ids, float* out,
               std::int64_t num_tokens, std::int64_t hidden_dim, std::int64_t vocab_size,
               cudaStream_t stream) {
  ENGINE_CHECK(table != nullptr, "embedding: table is null");
  ENGINE_CHECK(input_ids != nullptr, "embedding: input_ids is null");
  ENGINE_CHECK(out != nullptr, "embedding: out is null");
  ENGINE_CHECK(num_tokens >= 0, "embedding: num_tokens must be non-negative");
  ENGINE_CHECK(hidden_dim >= 0, "embedding: hidden_dim must be non-negative");
  ENGINE_CHECK(vocab_size > 0, "embedding: vocab_size must be positive");
  if (num_tokens == 0 || hidden_dim == 0) return;

  constexpr int THREADS = 256;
  const bool is_aligned_vec4 =
      (hidden_dim % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(table) % sizeof(float4) == 0) &&
      (reinterpret_cast<uintptr_t>(out) % sizeof(float4) == 0);

  if (is_aligned_vec4) {
    const int64_t num_vecs = hidden_dim / 4;
    dim3 grid(static_cast<unsigned int>((num_vecs + THREADS - 1) / THREADS),
              static_cast<unsigned int>(num_tokens));
    embedding_f32_vec4<<<grid, THREADS, 0, stream>>>(table, input_ids, out, num_vecs,
                                                     hidden_dim, vocab_size);
  } else {
    dim3 grid(static_cast<unsigned int>((hidden_dim + THREADS - 1) / THREADS),
              static_cast<unsigned int>(num_tokens));
    embedding_f32_scalar<<<grid, THREADS, 0, stream>>>(table, input_ids, out, hidden_dim,
                                                       vocab_size);
  }

  CUDA_CHECK_KERNEL();
}

void embedding_fp16(const half* table, const std::int32_t* input_ids, half* out,
                    std::int64_t num_tokens, std::int64_t hidden_dim,
                    std::int64_t vocab_size, cudaStream_t stream) {
  ENGINE_CHECK(table != nullptr, "embedding_fp16: table is null");
  ENGINE_CHECK(input_ids != nullptr, "embedding_fp16: input_ids is null");
  ENGINE_CHECK(out != nullptr, "embedding_fp16: out is null");
  ENGINE_CHECK(num_tokens >= 0, "embedding_fp16: num_tokens must be non-negative");
  ENGINE_CHECK(hidden_dim >= 0, "embedding_fp16: hidden_dim must be non-negative");
  ENGINE_CHECK(vocab_size > 0, "embedding_fp16: vocab_size must be positive");
  if (num_tokens == 0 || hidden_dim == 0) return;

  constexpr int THREADS = 256;
  const bool is_aligned_vec8 =
      (hidden_dim % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(table) % sizeof(uint4) == 0) &&
      (reinterpret_cast<uintptr_t>(out) % sizeof(uint4) == 0);

  if (is_aligned_vec8) {
    const int64_t num_vecs = hidden_dim / 8;
    dim3 grid(static_cast<unsigned int>((num_vecs + THREADS - 1) / THREADS),
              static_cast<unsigned int>(num_tokens));
    embedding_fp16_vec8<<<grid, THREADS, 0, stream>>>(table, input_ids, out, num_vecs,
                                                      hidden_dim, vocab_size);
  } else {
    dim3 grid(static_cast<unsigned int>((hidden_dim + THREADS - 1) / THREADS),
              static_cast<unsigned int>(num_tokens));
    embedding_fp16_scalar<<<grid, THREADS, 0, stream>>>(table, input_ids, out, hidden_dim,
                                                        vocab_size);
  }

  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
