#pragma once
//===----------------------------------------------------------------------===//
// engine/cpu_ref.hpp -- CPU reference implementations.
//
// These are the ORACLE for the whole project. They are written for obvious
// correctness, not speed: straightforward loops, FP64 accumulators where it
// matters, no blocking, no SIMD, no threads. Do not optimise them. Their entire
// job is to be so simple that they are self-evidently right, so that when a CUDA
// kernel disagrees with them you know the kernel is wrong.
//
// Every function here is also the specification for the corresponding CUDA kernel
// in engine/kernels.hpp -- same name, same argument order, same memory layout.
//
// MEMORY LAYOUT CONVENTION (applies to the entire project):
//   All 2-D data is ROW-MAJOR and contiguous. Element (r, c) of an R x C buffer
//   lives at index r * C + c. Getting this wrong is the single most common source
//   of matmul bugs, so it is stated once here and never varied.
//===----------------------------------------------------------------------===//

#include <cstdint>

namespace engine::cpu {

/// out[i] = a[i] + b[i]
void vector_add(const float* a, const float* b, float* out, std::int64_t n);

/// Returns sum(x[0..n)). Accumulates in double to keep the oracle trustworthy for
/// large n -- a float accumulator loses low-order bits and would make the CUDA
/// kernel look wrong when it is actually more accurate.
double reduce_sum(const float* x, std::int64_t n);

/// Row-wise softmax of a `rows` x `cols` row-major matrix.
///
/// Uses the max-subtraction trick:  softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)
/// where m = max(x). Mathematically identical to the naive form, but without it
/// exp() overflows to inf for inputs above ~88.7 in FP32. Every real softmax does
/// this, and the online (single-pass) version of this same trick is the core idea
/// of FlashAttention -- see docs/02-cuda-exercises.md, exercise 3.
void softmax_rows(const float* in, float* out, std::int64_t rows, std::int64_t cols);

/// Row-wise RMSNorm:  out[r][c] = in[r][c] / sqrt(mean(in[r][:]^2) + eps) * weight[c]
///
/// This is the normalisation LLaMA uses. Note what is ABSENT compared to LayerNorm:
/// no mean subtraction and no learned bias. That makes it cheaper (one reduction
/// instead of two) and is why LLaMA-family models use it.
///
/// `weight` is a per-channel gain of length `cols`. Pass nullptr for no scaling.
void rmsnorm(const float* in, const float* weight, float* out, std::int64_t rows,
             std::int64_t cols, float eps);

/// C = A * B  where A is M x K, B is K x N, C is M x N. All row-major.
/// Accumulates in double so the oracle is not itself a source of error.
void matmul(const float* A, const float* B, float* C, std::int64_t M, std::int64_t N,
            std::int64_t K);

//===----------------------------------------------------------------------===//
// Module 3 — Fused operations.
//
// Each of these computes the same result as calling two separate functions in
// sequence.  The oracle deliberately does them in two passes (no fusion) so
// that it stays "obviously correct" — the fusion is the kernel's job.
//===----------------------------------------------------------------------===//

/// Fused residual-add + RMSNorm (Exercise 8).
///
/// Computes:
///   sum_out[r][c]  = x[r][c] + residual[r][c]
///   norm_out[r][c] = sum_out[r][c] / sqrt(mean(sum_out[r][:]^2) + eps) * weight[c]
///
/// Both outputs are written.  sum_out is needed by the next residual connection;
/// norm_out feeds into the next sub-layer.  `weight` may be nullptr (no scaling).
void residual_rmsnorm(const float* x, const float* residual,
                      const float* weight, float* norm_out, float* sum_out,
                      std::int64_t rows, std::int64_t cols, float eps);

/// Fused RMSNorm + linear projection (Exercise 7).
///
/// Computes:
///   temp[r][k] = in[r][k] / sqrt(mean(in[r][:]^2) + eps) * rms_weight[k]
///   out[r][n]  = sum_k temp[r][k] * W[k][n]
///
/// The intermediate `temp` is never materialised in global memory.
/// `rms_weight` may be nullptr (no per-channel gain).
void rmsnorm_linear(const float* in, const float* rms_weight,
                    const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K, float eps);

}  // namespace engine::cpu
