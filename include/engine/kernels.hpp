#pragma once
//===----------------------------------------------------------------------===//
// engine/kernels.hpp -- host-side launch API for the CUDA kernels.
//
// DESIGN DECISION: these take RAW POINTERS and explicit dimensions, not Tensor.
//
// Two reasons, and they both matter:
//
//   1. Scheduling. Kernels are your August/December work; Tensor is October's. If
//      kernels took Tensor, you could not test a single kernel until Tensor was
//      finished. Raw pointers break that dependency, so the two modules progress
//      independently.
//   2. It is the right layering anyway. cuBLAS, cuDNN and llama.cpp's ggml all
//      expose pointer+dims at the kernel boundary and keep the tensor abstraction
//      strictly above it. A kernel should not know what a Tensor is.
//
// Tensor-aware overloads will be added later as thin wrappers that unpack .data()
// and .shape() and forward to these. See docs/adr/0003.
//
// Every function here mirrors a CPU reference in engine/cpu_ref.hpp with the same
// name and argument order. That is what makes the test harness a simple A/B compare.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#if ENGINE_HAS_CUDA

#include <engine/half.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace engine::cuda {

using engine::half;

//===----------------------------------------------------------------------===//
// All pointers below are DEVICE pointers. Passing a host pointer will not fail
// at compile time -- it will fault at runtime, or worse, silently read garbage.
// Use engine::DeviceBuffer<T> (engine/device_buffer.hpp) so the type system helps.
//
// `stream = 0` is the default (legacy) stream. Streams become relevant when you
// overlap H2D copies with compute; until then 0 is correct and simplest.
//===----------------------------------------------------------------------===//

/// Exercise 1 (WORKED EXAMPLE -- read kernels/vector_add.cu first).
/// out[i] = a[i] + b[i]
void vector_add(const float* a, const float* b, float* out, std::int64_t n,
                cudaStream_t stream = 0);

/// Exercise 2. Sum-reduce `n` elements into a single float.
/// `out` must point to at least 1 float of DEVICE memory.
void reduce_sum(const float* x, float* out, std::int64_t n, cudaStream_t stream = 0);

/// Exercise 3. Row-wise softmax of a rows x cols row-major matrix.
/// Must be numerically stable (subtract the row max) -- see cpu_ref for why.
void softmax_rows(const float* in, float* out, std::int64_t rows, std::int64_t cols,
                  cudaStream_t stream = 0);

/// Exercise 4. Row-wise RMSNorm. `weight` may be nullptr.
void rmsnorm(const float* in, const float* weight, float* out, std::int64_t rows,
             std::int64_t cols, float eps, cudaStream_t stream = 0);

/// Exercise 5. C[MxN] = A[MxK] * B[KxN], row-major. One thread per output element.
/// This is the SLOW baseline you will beat -- keep it forever as a benchmark point.
void matmul_naive(const float* A, const float* B, float* C, std::int64_t M,
                  std::int64_t N, std::int64_t K, cudaStream_t stream = 0);

/// Exercise 6. Same maths as matmul_naive, but tiled through shared memory.
/// Target: a large speedup over naive at M=N=K=1024. Explaining *why* it is faster
/// (arithmetic intensity / global-memory traffic reduced by a factor of TILE) is
/// the actual deliverable here.
void matmul_tiled(const float* A, const float* B, float* C, std::int64_t M,
                  std::int64_t N, std::int64_t K, cudaStream_t stream = 0);

/// Advanced Phase 4 GEMM: Register-tiled 2D GEMM.
/// Uses 128x128 block tiles, 8x8 thread register tiles, float4 vectorized loads,
/// and outer-product math to bypass shared-memory bandwidth saturation.
void matmul_register_tiled(const float* A, const float* B, float* C, std::int64_t M,
                           std::int64_t N, std::int64_t K, cudaStream_t stream = 0);

/// GEMV: Matrix-vector multiply out[N] = x[K] * A[K x N], row-major.
/// Specialized for M=1 decode-time token generation to bypass 2D tile waste.
void gemv(const float* A, const float* x, float* out, std::int64_t N, std::int64_t K,
          cudaStream_t stream = 0);

//===----------------------------------------------------------------------===//
// Module 3 — Fused operations.
//===----------------------------------------------------------------------===//

/// Exercise 7. Fused RMSNorm + Linear projection.
/// out[M x N] = RMSNorm(in[M x K], rms_weight[K], eps) * W[K x N]
/// Eliminates the intermediate M×K write/read between RMSNorm and matmul.
/// `rms_weight` may be nullptr (no per-channel gain).
/// Dynamically dispatches fused 1D kernel for M=1 (decode) and tiled matmul for M>1 (prefill).
void rmsnorm_linear(const float* in, const float* rms_weight, const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K, float eps,
                    cudaStream_t stream = 0);

/// Direct execution of the 1D fused kernel regardless of M (used for benchmarking/profiling).
void rmsnorm_linear_fused_direct(const float* in, const float* rms_weight, const float* W,
                                 float* out, std::int64_t M, std::int64_t N,
                                 std::int64_t K, float eps, cudaStream_t stream = 0);

/// Exercise 8. Fused Residual-Add + RMSNorm.
/// norm_out[r][c] = RMSNorm(x[r][c] + residual[r][c], weight, eps)
/// sum_out[r][c]  = x[r][c] + residual[r][c]   (for the next residual connection)
/// `weight` may be nullptr (no scaling).
void residual_rmsnorm(const float* x, const float* residual, const float* weight,
                      float* norm_out, float* sum_out, std::int64_t rows,
                      std::int64_t cols, float eps, cudaStream_t stream = 0);

//===----------------------------------------------------------------------===//
// Phase 3 — FP16 and missing inference kernels.
//===----------------------------------------------------------------------===//

/// Dense matrix-vector product in FP16: out[N] = x[K] * A[K x N], row-major.
/// Specialized for decode token generation using 128-bit vector memory instructions.
void gemv_fp16(const half* A, const half* x, half* out, std::int64_t N, std::int64_t K,
               cudaStream_t stream = 0);

/// Embedding lookup: gathers rows from `table` into `out` according to `input_ids`.
/// table: [vocab_size, hidden_dim], input_ids: [num_tokens], out: [num_tokens, hidden_dim]
void embedding(const float* table, const std::int32_t* input_ids, float* out,
               std::int64_t num_tokens, std::int64_t hidden_dim, std::int64_t vocab_size,
               cudaStream_t stream = 0);
void embedding_fp16(const half* table, const std::int32_t* input_ids, half* out,
                    std::int64_t num_tokens, std::int64_t hidden_dim,
                    std::int64_t vocab_size, cudaStream_t stream = 0);

/// Argmax / Greedy sampling: finds the index of the maximum logit and writes to out_token.
void argmax(const float* logits, std::int32_t* out_token, std::int64_t vocab_size,
            cudaStream_t stream = 0);
void argmax_fp16(const half* logits, std::int32_t* out_token, std::int64_t vocab_size,
                 cudaStream_t stream = 0);

/// SwiGLU activation: out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
void swiglu(const float* gate, const float* up, float* out, std::int64_t n,
            cudaStream_t stream = 0);
void swiglu_fp16(const half* gate, const half* up, half* out, std::int64_t n,
                 cudaStream_t stream = 0);

}  // namespace engine::cuda

#endif  // ENGINE_HAS_CUDA
