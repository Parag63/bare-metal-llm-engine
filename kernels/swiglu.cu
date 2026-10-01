//===----------------------------------------------------------------------===//
// kernels/swiglu.cu -- Fused SwiGLU activation kernel for LLaMA-style MLPs.
//
// Computes out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
//
// FUSION BENEFITS:
// An unfused implementation requires two separate kernel launches:
//   1. silu_kernel: reads gate (4N bytes), writes silu_out (4N bytes) to DRAM.
//   2. mul_kernel:  reads silu_out (4N bytes) and up (4N bytes), writes out (4N bytes).
//   Total DRAM traffic: 20N bytes.
//
// The fused kernel combines SiLU and elementwise multiplication into a single pass:
//   Reads gate (4N bytes) and up (4N bytes), computes in registers, writes out (4N bytes).
//   Total DRAM traffic: 12N bytes (40% reduction in memory traffic + eliminates 1 launch).
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/kernels.hpp>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

__device__ inline float silu_f(float x) {
  return x / (1.0f + __expf(-x));
}

// Vectorized FP32 kernel: each thread processes 4 floats (128 bits)
__global__ void swiglu_f32_vec4(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ out,
                                int64_t num_vecs) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_vecs) return;

  const float4 g = reinterpret_cast<const float4*>(gate)[idx];
  const float4 u = reinterpret_cast<const float4*>(up)[idx];

  float4 r;
  r.x = silu_f(g.x) * u.x;
  r.y = silu_f(g.y) * u.y;
  r.z = silu_f(g.z) * u.z;
  r.w = silu_f(g.w) * u.w;

  reinterpret_cast<float4*>(out)[idx] = r;
}

// Fallback scalar FP32 kernel
__global__ void swiglu_f32_scalar(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out,
                                  int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  out[idx] = silu_f(gate[idx]) * up[idx];
}

// Vectorized FP16 kernel: each thread processes 8 halfs (128 bits = uint4)
__global__ void swiglu_fp16_vec8(const half* __restrict__ gate,
                                 const half* __restrict__ up,
                                 half* __restrict__ out,
                                 int64_t num_vecs) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_vecs) return;

  const uint4 g_raw = reinterpret_cast<const uint4*>(gate)[idx];
  const uint4 u_raw = reinterpret_cast<const uint4*>(up)[idx];

  const half* g = reinterpret_cast<const half*>(&g_raw);
  const half* u = reinterpret_cast<const half*>(&u_raw);

  half r_half[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float gf = __half2float(g[i]);
    const float uf = __half2float(u[i]);
    r_half[i] = __float2half(silu_f(gf) * uf);
  }

  reinterpret_cast<uint4*>(out)[idx] = *reinterpret_cast<const uint4*>(r_half);
}

// Fallback scalar FP16 kernel
__global__ void swiglu_fp16_scalar(const half* __restrict__ gate,
                                   const half* __restrict__ up,
                                   half* __restrict__ out,
                                   int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  const float gf = __half2float(gate[idx]);
  const float uf = __half2float(up[idx]);
  out[idx] = __float2half(silu_f(gf) * uf);
}

}  // namespace

void swiglu(const float* gate, const float* up, float* out, std::int64_t n, cudaStream_t stream) {
  ENGINE_CHECK(gate != nullptr, "swiglu: gate is null");
  ENGINE_CHECK(up != nullptr, "swiglu: up is null");
  ENGINE_CHECK(out != nullptr, "swiglu: out is null");
  ENGINE_CHECK(n >= 0, "swiglu: n must be non-negative");
  if (n == 0) return;

  constexpr int THREADS = 256;
  const bool is_aligned_vec4 = (n % 4 == 0) &&
                               (reinterpret_cast<uintptr_t>(gate) % sizeof(float4) == 0) &&
                               (reinterpret_cast<uintptr_t>(up) % sizeof(float4) == 0) &&
                               (reinterpret_cast<uintptr_t>(out) % sizeof(float4) == 0);

  if (is_aligned_vec4) {
    const int64_t num_vecs = n / 4;
    const int blocks = static_cast<int>((num_vecs + THREADS - 1) / THREADS);
    swiglu_f32_vec4<<<blocks, THREADS, 0, stream>>>(gate, up, out, num_vecs);
  } else {
    const int blocks = static_cast<int>((n + THREADS - 1) / THREADS);
    swiglu_f32_scalar<<<blocks, THREADS, 0, stream>>>(gate, up, out, n);
  }

  CUDA_CHECK_KERNEL();
}

void swiglu_fp16(const half* gate, const half* up, half* out, std::int64_t n, cudaStream_t stream) {
  ENGINE_CHECK(gate != nullptr, "swiglu_fp16: gate is null");
  ENGINE_CHECK(up != nullptr, "swiglu_fp16: up is null");
  ENGINE_CHECK(out != nullptr, "swiglu_fp16: out is null");
  ENGINE_CHECK(n >= 0, "swiglu_fp16: n must be non-negative");
  if (n == 0) return;

  constexpr int THREADS = 256;
  const bool is_aligned_vec8 = (n % 8 == 0) &&
                               (reinterpret_cast<uintptr_t>(gate) % sizeof(uint4) == 0) &&
                               (reinterpret_cast<uintptr_t>(up) % sizeof(uint4) == 0) &&
                               (reinterpret_cast<uintptr_t>(out) % sizeof(uint4) == 0);

  if (is_aligned_vec8) {
    const int64_t num_vecs = n / 8;
    const int blocks = static_cast<int>((num_vecs + THREADS - 1) / THREADS);
    swiglu_fp16_vec8<<<blocks, THREADS, 0, stream>>>(gate, up, out, num_vecs);
  } else {
    const int blocks = static_cast<int>((n + THREADS - 1) / THREADS);
    swiglu_fp16_scalar<<<blocks, THREADS, 0, stream>>>(gate, up, out, n);
  }

  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
