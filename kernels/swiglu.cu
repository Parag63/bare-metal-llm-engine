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

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::cuda {

namespace {

__device__ __forceinline__ float silu_f(float x) {
  return x * __fdividef(1.0f, 1.0f + __expf(-x));
}

// Vectorized FP32 kernel: each thread processes 4 floats (128 bits) via grid-stride loop
__global__ void swiglu_f32_vec4(const float* __restrict__ gate,
                                const float* __restrict__ up, float* __restrict__ out,
                                int64_t num_vecs) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  const float4* __restrict__ g_vec = reinterpret_cast<const float4*>(gate);
  const float4* __restrict__ u_vec = reinterpret_cast<const float4*>(up);
  float4* __restrict__ o_vec = reinterpret_cast<float4*>(out);

  // 4x unroll for instruction-level parallelism and maximum memory controller saturation
  for (; idx + 3 * stride < num_vecs; idx += 4 * stride) {
    const float4 g0 = g_vec[idx];
    const float4 g1 = g_vec[idx + stride];
    const float4 g2 = g_vec[idx + 2 * stride];
    const float4 g3 = g_vec[idx + 3 * stride];

    const float4 u0 = u_vec[idx];
    const float4 u1 = u_vec[idx + stride];
    const float4 u2 = u_vec[idx + 2 * stride];
    const float4 u3 = u_vec[idx + 3 * stride];

    float4 r0, r1, r2, r3;
    r0.x = silu_f(g0.x) * u0.x;
    r0.y = silu_f(g0.y) * u0.y;
    r0.z = silu_f(g0.z) * u0.z;
    r0.w = silu_f(g0.w) * u0.w;

    r1.x = silu_f(g1.x) * u1.x;
    r1.y = silu_f(g1.y) * u1.y;
    r1.z = silu_f(g1.z) * u1.z;
    r1.w = silu_f(g1.w) * u1.w;

    r2.x = silu_f(g2.x) * u2.x;
    r2.y = silu_f(g2.y) * u2.y;
    r2.z = silu_f(g2.z) * u2.z;
    r2.w = silu_f(g2.w) * u2.w;

    r3.x = silu_f(g3.x) * u3.x;
    r3.y = silu_f(g3.y) * u3.y;
    r3.z = silu_f(g3.z) * u3.z;
    r3.w = silu_f(g3.w) * u3.w;

    o_vec[idx] = r0;
    o_vec[idx + stride] = r1;
    o_vec[idx + 2 * stride] = r2;
    o_vec[idx + 3 * stride] = r3;
  }

  // Remainder loop
  for (; idx < num_vecs; idx += stride) {
    const float4 g = g_vec[idx];
    const float4 u = u_vec[idx];

    float4 r;
    r.x = silu_f(g.x) * u.x;
    r.y = silu_f(g.y) * u.y;
    r.z = silu_f(g.z) * u.z;
    r.w = silu_f(g.w) * u.w;

    o_vec[idx] = r;
  }
}

// Fallback scalar FP32 kernel
__global__ void swiglu_f32_scalar(const float* __restrict__ gate,
                                  const float* __restrict__ up, float* __restrict__ out,
                                  int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += stride) {
    out[idx] = silu_f(gate[idx]) * up[idx];
  }
}

// Vectorized FP16 kernel: each thread processes 8 halfs (128 bits = uint4) via grid-stride loop
__global__ void swiglu_fp16_vec8(const half* __restrict__ gate,
                                 const half* __restrict__ up, half* __restrict__ out,
                                 int64_t num_vecs) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  const uint4* __restrict__ g_vec = reinterpret_cast<const uint4*>(gate);
  const uint4* __restrict__ u_vec = reinterpret_cast<const uint4*>(up);
  uint4* __restrict__ o_vec = reinterpret_cast<uint4*>(out);

  // 2x unroll (16 halfs per thread per iteration)
  for (; idx + stride < num_vecs; idx += 2 * stride) {
    const uint4 g0_raw = g_vec[idx];
    const uint4 g1_raw = g_vec[idx + stride];
    const uint4 u0_raw = u_vec[idx];
    const uint4 u1_raw = u_vec[idx + stride];

    const half* g0 = reinterpret_cast<const half*>(&g0_raw);
    const half* g1 = reinterpret_cast<const half*>(&g1_raw);
    const half* u0 = reinterpret_cast<const half*>(&u0_raw);
    const half* u1 = reinterpret_cast<const half*>(&u1_raw);

    half r0[8], r1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      r0[i] = __float2half(silu_f(__half2float(g0[i])) * __half2float(u0[i]));
      r1[i] = __float2half(silu_f(__half2float(g1[i])) * __half2float(u1[i]));
    }
    o_vec[idx] = *reinterpret_cast<const uint4*>(r0);
    o_vec[idx + stride] = *reinterpret_cast<const uint4*>(r1);
  }

  // Remainder loop
  for (; idx < num_vecs; idx += stride) {
    const uint4 g_raw = g_vec[idx];
    const uint4 u_raw = u_vec[idx];
    const half* g = reinterpret_cast<const half*>(&g_raw);
    const half* u = reinterpret_cast<const half*>(&u_raw);
    half r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      r[i] = __float2half(silu_f(__half2float(g[i])) * __half2float(u[i]));
    }
    o_vec[idx] = *reinterpret_cast<const uint4*>(r);
  }
}

// Fallback scalar FP16 kernel
__global__ void swiglu_fp16_scalar(const half* __restrict__ gate,
                                   const half* __restrict__ up, half* __restrict__ out,
                                   int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += stride) {
    const float gf = __half2float(gate[idx]);
    const float uf = __half2float(up[idx]);
    out[idx] = __float2half(silu_f(gf) * uf);
  }
}

}  // namespace

void swiglu(const float* gate, const float* up, float* out, std::int64_t n,
            cudaStream_t stream) {
  ENGINE_CHECK(gate != nullptr, "swiglu: gate is null");
  ENGINE_CHECK(up != nullptr, "swiglu: up is null");
  ENGINE_CHECK(out != nullptr, "swiglu: out is null");
  ENGINE_CHECK(n >= 0, "swiglu: n must be non-negative");
  if (n == 0) return;

  constexpr int THREADS = 256;
  const bool is_aligned_vec4 =
      (n % 4 == 0) && (reinterpret_cast<uintptr_t>(gate) % sizeof(float4) == 0) &&
      (reinterpret_cast<uintptr_t>(up) % sizeof(float4) == 0) &&
      (reinterpret_cast<uintptr_t>(out) % sizeof(float4) == 0);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  const int64_t blocks_wanted = static_cast<int64_t>(num_sms) * 32;

  if (is_aligned_vec4) {
    const int64_t num_vecs = n / 4;
    const int64_t blocks_needed = (num_vecs + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(std::min(blocks_needed, blocks_wanted));
    swiglu_f32_vec4<<<blocks, THREADS, 0, stream>>>(gate, up, out, num_vecs);
  } else {
    const int64_t blocks_needed = (n + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(std::min(blocks_needed, blocks_wanted));
    swiglu_f32_scalar<<<blocks, THREADS, 0, stream>>>(gate, up, out, n);
  }

  CUDA_CHECK_KERNEL();
}

void swiglu_fp16(const half* gate, const half* up, half* out, std::int64_t n,
                 cudaStream_t stream) {
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

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  const int64_t blocks_wanted = static_cast<int64_t>(num_sms) * 32;

  if (is_aligned_vec8) {
    const int64_t num_vecs = n / 8;
    const int64_t blocks_needed = (num_vecs + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(std::min(blocks_needed, blocks_wanted));
    swiglu_fp16_vec8<<<blocks, THREADS, 0, stream>>>(gate, up, out, num_vecs);
  } else {
    const int64_t blocks_needed = (n + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(std::min(blocks_needed, blocks_wanted));
    swiglu_fp16_scalar<<<blocks, THREADS, 0, stream>>>(gate, up, out, n);
  }

  CUDA_CHECK_KERNEL();
}

}  // namespace engine::cuda
