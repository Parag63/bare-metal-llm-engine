# ADR 0006 — Kernel Fusion Strategy for Transformer Layers

**Status:** accepted · **Date:** 2026-09-29 · **Applies to:** `kernels/rmsnorm_linear.cu`, `kernels/residual_rmsnorm.cu`, `include/engine/kernels.hpp`

## Context

In modern transformer language models (LLaMA-7B, Mistral, TinyLlama), inference latency is
dominated by memory bandwidth and launch overhead rather than peak arithmetic capability.
On an NVIDIA RTX 4090, the theoretical balance point is approximately 82 FLOP/byte
(82.6 TFLOP/s FP32 vs. 1,008 GB/s memory bandwidth). Any operation with an arithmetic
intensity below this ratio is memory-bandwidth bound.

Every standard transformer layer in LLaMA-7B executes two structural sub-layer boundaries:
1. **Pre-sublayer:** `RMSNorm → Linear` (hidden → RMSNorm → QKV projection, and hidden → RMSNorm → gate/up projection).
2. **Post-sublayer:** `Residual Add → RMSNorm` (post-attention residual add → pre-FFN RMSNorm, and post-FFN residual add → next-layer RMSNorm).

In an unfused execution pipeline:
- **`RMSNorm → Linear`**: RMSNorm writes the full normalized activation matrix ($M \times K$, 16 KiB at $M=1$ or 8 MiB at $M=512$) to DRAM. The subsequent GEMM kernel reads that exact activation matrix right back from DRAM.
- **`Residual Add → RMSNorm`**: Residual add reads $x$ and $residual$, writing the sum to DRAM (12 bytes/element moved). RMSNorm immediately reads the sum back from DRAM and writes the normalized output (8 bytes/element moved). Total DRAM traffic: 20 bytes/element.
- **Launch overhead:** Each kernel launch consumes ~3–5 µs of CPU/driver dispatch time. For a 32-layer model, back-to-back unfused execution requires 64 launches per token where 32 would do, incurring ~320 µs/token of pure overhead before arithmetic starts.

## Decision

We implement two targeted fused kernels in Module 3:

1. **`rmsnorm_linear` (Exercise 7):**
   Fuses row-wise RMSNorm with the subsequent linear projection matrix multiplication ($C = \text{RMSNorm}(A) \cdot W$).
   - The row is loaded into shared memory and normalized in-place via a cooperative block reduction.
   - The normalized activations in shared memory directly feed the matrix multiply accumulation loop, broadcasting to threads across each warp with zero shared memory bank conflicts.
   - The intermediate normalized activation matrix is **never written to DRAM**.

2. **`residual_rmsnorm` (Exercise 8):**
   Fuses elementwise residual addition with RMSNorm.
   - In a single pass over $x$ and $residual$, threads compute the elementwise sum $s = x[c] + residual[c]$.
   - The sum $s$ is written directly to `sum_out` (which is needed by the residual stream for the next residual addition) and its square is accumulated into thread registers.
   - Following a block reduction and scale calculation, normalized values are written to `norm_out`.
   - DRAM traffic is reduced from 20 bytes/element to 16 bytes/element (saving 1 full global memory read pass).

Both kernels adhere strictly to ADR 0003: they expose raw device pointers, explicit 64-bit integer dimensions, and an optional `cudaStream_t`.

## Alternatives Considered and Deferred

- **FlashAttention (tiled online softmax + GEMM fusion):**
  Deferred to Module 4. FlashAttention fuses the attention matrix computation ($Q K^T$), scaling, row-wise online softmax, and value projection ($P V$) into a single tiled SRAM schedule. Module 3 intentionally focuses on 1D strip and pre-/post-sublayer fusions to master the memory-schedule paradigm before tackling multi-dimensional online softmax fusion.
- **SwiGLU FFN activation fusion:**
  Fusing the gate and up projections with the SiLU elementwise multiplication ($\text{silu}(x W_{gate}) \cdot (x W_{up})$) is a high-value optimization for LLaMA-style feed-forward networks. This is scheduled for Module 5 alongside model weight mapping.
- **End-to-End monolithic layer fusion:**
  Fusing an entire transformer layer into a single megakernel was rejected. Monolithic kernels cause severe register spilling, lower warp occupancy, and hinder modular debugging. Two-operator fusions strike the optimal balance between eliminating memory traffic and preserving hardware occupancy.

## Consequences

- **Bandwidth reduction:** For prefill batches ($M=512, K=4096$), `rmsnorm_linear` eliminates 8 MiB of write traffic and 8 MiB of read traffic (16 MiB total) per projection. `residual_rmsnorm` eliminates 8 MiB of read traffic per sub-layer.
- **Launch latency:** Reduces kernel launches by 64 launches per token across a 32-layer transformer, reclaiming up to ~320 µs of driver overhead per token during autoregressive decode.
- **Shared memory footprint:** `rmsnorm_linear` allocates $(K + kBlockSize) \times 4$ bytes of dynamic shared memory. For LLaMA-7B ($K=4096, kBlockSize=256$), this requires 17.2 KiB per block, well within the default 48 KiB hardware limit on modern NVIDIA architectures (sm_89 provides up to 100 KiB).
- **Verification integrity:** Because the CPU oracles (`src/cpu_ref/rmsnorm_linear_cpu.cpp`, `src/cpu_ref/residual_rmsnorm_cpu.cpp`) implement the exact mathematical equivalent using standard two-pass loops, the test suite can independently verify both fused correctness against PyTorch/NumPy reference data and assert equivalence against the composing separate kernels (`rmsnorm` + `matmul_tiled`).
