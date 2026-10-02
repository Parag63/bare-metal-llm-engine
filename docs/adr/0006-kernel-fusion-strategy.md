# ADR 0006 — Kernel Fusion Strategy for Transformer Layers

**Status:** accepted · **Date:** 2026-09-29 · **Applies to:** `kernels/rmsnorm_linear.cu`, `kernels/residual_rmsnorm.cu`, `include/engine/kernels.hpp`

## Context

In modern transformer language models (LLaMA-7B, Mistral, TinyLlama), inference latency is
dominated by memory bandwidth and launch overhead rather than peak arithmetic capability.
On an NVIDIA GeForce RTX 4070 SUPER (Ada Lovelace, sm_89), the theoretical balance point is
approximately 70.4 FLOP/byte (35.5 TFLOP/s FP32 vs. 504 GB/s GDDR6X memory bandwidth).
Any operation with an arithmetic intensity below this ratio is memory-bandwidth bound.

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

---

## Addendum (October 2026): Empirical Defeat of 1D Strip Fusion and Dynamic Dispatch

Subsequent empirical benchmarking and profiling with Nsight Compute revealed that the naive 1D strip fusion implementation (`rmsnorm_linear_kernel` / `rmsnorm_linear_fused_direct`) is defeated by separate specialized kernels at **both** operational shapes:

1. **Prefill Defeat ($M=512$):**
   - **Hypothesis:** Eliminating the 16 MiB intermediate normalized activation roundtrip was expected to produce a 15–30% speedup.
   - **Measurement:** 1D fused took **13.56 ms** vs. **0.99 ms** for `rmsnorm + matmul_register_tiled` — a **$13.6\times$ defeat**.
   - **Root Cause:** In 1D row fusion, each block processes one row independently to keep normalized activations in shared memory. This destroys 2D register tiling across tokens ($M$) and forces the weight matrix $W$ to be reloaded from DRAM for every token. As measured by Nsight Compute (`ncu`), DRAM read traffic exploded from 0.82 GB to 3.37 GB (+2.55 GB reload penalty). The 16 MiB activation savings is dwarfed $160\times$ over by the weight reload penalty.

2. **Decode Defeat ($M=1$):**
   - **Hypothesis:** An initial audit claimed a "+43% win" for decode. However, this compared against a strawman baseline (`rmsnorm + matmul_tiled` at $M=1$, where $31/32$ threads are idle padding, running at 0.52 ms).
   - **Measurement:** Against the fair baseline (`rmsnorm + gemv`), separate execution runs in **0.149 ms** (465 GB/s, 92.3% peak BW), whereas 1D fused runs in **0.366 ms** (182 GB/s) — a **$2.45\times$ defeat**.
   - **Root Cause:** At $M=1$ and $N=4096$, 1D strip fusion launches only $\lceil 4096 / 256 \rceil = 16$ blocks. On the RTX 4070 SUPER (56 SMs), 40 SMs (71% of the GPU) sit completely idle. Furthermore, each thread performs an unvectorized scalar loop over $K$. In contrast, `gemv` launches 64 blocks of 8 warps, uses `float2` vector loads, and performs cooperative multi-warp tree reductions across $K$.

### Production Resolution: Dynamic Winning-Path Dispatch

Recognizing this empirical negative result, the public engine entry point `engine::cuda::rmsnorm_linear` does not execute the defeated 1D fused kernel. Instead, it dynamically dispatches to the winning paths:
- **$M = 1$ (Decode):** Dispatches to `rmsnorm` followed by `gemv` (0.149 ms, 465 GB/s).
- **$M > 1$ (Prefill):** Dispatches to `rmsnorm` followed by `matmul_register_tiled` (0.99 ms, 17.35 TFLOP/s).
- **Zero Allocation Overhead:** Intermediate workspaces are allocated from the high-throughput `PoolAllocator(Device::CUDA)` slab allocator, eliminating `cudaMalloc` driver roundtrips.

### Residual RMSNorm Cold DRAM Speedup Verification

For `residual_rmsnorm`, intermediate activations do not reload weights. In cold DRAM streaming at $4096 \times 4096$ (268.4 MB total working set, exceeding the 48 MB L2 cache):
- **Separate (`vector_add + rmsnorm`):** 5 tensor passes ($335.5\text{ MB}$) in **0.763 ms** (439.7 GB/s).
- **Fused (`residual_rmsnorm`):** 4 tensor passes ($268.4\text{ MB}$) in **0.618 ms** (434.1 GB/s).
- **Achieved Speedup:** $0.763 / 0.618 = \mathbf{1.235\times}$ (measured) / $\mathbf{1.246\times}$ (best min), perfectly matching the theoretical memory traffic ceiling of $5/4 = \mathbf{1.25\times}$ (+25%).

