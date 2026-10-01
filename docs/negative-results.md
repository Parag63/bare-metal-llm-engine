# Negative Results & Non-Optimizations

> "The point of this project is not merely to report fast numbers, but to understand why optimizations succeed or fail on real hardware. An optimization that produces zero speedup is just as informative as one that yields 2×, provided you can measure and explain the difference."
> — Engineering Lab Principle

This document records architectural optimizations and techniques that were evaluated, implemented, or considered in **bare-metal-llm-engine**, but yielded negligible speedup, caused regressions, or proved counterproductive on our target hardware (**NVIDIA Ada Lovelace, sm_89, RTX 4070 SUPER / RTX 4090**).

---

## 1. Shared-Memory Bank Conflict Padding (`[TILE][TILE+1]`) in Tiled Matmul

### The Hypothesis
On classical CUDA architectures, shared memory is divided into 32 banks (4 bytes wide each). Successive 32-bit words map to successive banks.
In matrix multiplication:
- Thread $(ty, tx)$ reads $A[ty][k]$ and $B[k][tx]$.
- While threads in a warp access row $k$ of tile $B$ simultaneously (same bank = broadcast, collision-free), accessing column $k$ across different rows could cause multiple threads in the warp to hit the same memory bank with different addresses, resulting in a multi-way bank conflict (serializing loads).
- The standard textbook optimization is to pad the shared memory allocation:
  ```cpp
  __shared__ float As[32][33];  // +1 column stride to skew bank alignment
  __shared__ float Bs[32][33];
  ```

### The Measurement & Finding
On Ada Lovelace (sm_89):
- **Unpadded (`[32][32]`):** 2,563 GFLOP/s at $4096^3$
- **Padded (`[32][33]`):** 2,558 GFLOP/s at $4096^3$ ($\Delta = -0.2\%$, within measurement noise)

### Why It Failed to Help
1. **Lovelace Warp Scheduler & Shared-Memory Load Datapaths:** The Ada memory subsystem has high-throughput shared-memory crossbars and efficient multicast/broadcast units.
2. **Address Arithmetic Overhead:** Indexing a non-power-of-2 stride (`33 * sizeof(float)`) requires integer multiplication or additional index calculation instructions in the inner loop rather than simple bit-shifts (`ty << 5`), offsetting any negligible shared-memory latency savings.
3. **Register Reuse Dominates:** The inner tile loop accumulates into registers. Shared memory is only accessed once per inner step. Bank latency is already hidden by the arithmetic instructions in the register pipeline.

---

## 2. Kernel Fusion at Decode Shapes ($M=1$) for Cache-Resident Intermediates

### The Hypothesis
In Module 3, fusing `RMSNorm + Linear` (`rmsnorm_linear`) eliminated the intermediate normalized activation tensor $X_{\text{norm}}$, predicting a DRAM savings of:
$$\Delta \text{Bytes} = 2 \times M \times K \times 4\text{ bytes}$$
At prefill batch size $M=512, K=4096$:
$$\Delta \text{Bytes} = 2 \times 512 \times 4096 \times 4 = 16\text{ MiB DRAM eliminated}$$
At decode token generation ($M=1, K=4096$), we hypothesized that fusing would similarly eliminate DRAM traffic.

### The Measurement & Finding
- **$M=512, K=4096, N=4096$:** Separate = 7.10 ms, Fused = 13.62 ms (Note: GEMM tile decomposition at $M=512$ with fused normalization requires specialized prefill tiling).
- **$M=1, K=4096, N=4096$:** Separate = 0.553 ms, Fused = 0.378 ms ($1.46\times$ speedup).
- **$M=1, K=4096, N=12288$ (MLP Gate/Up projection):** Separate = 1.23 ms, Fused = 0.488 ms ($2.52\times$ speedup).

### Why the Dramatic Difference Occurs
At $M=1, K=4096$:
The intermediate vector $X_{\text{norm}}$ is only:
$$1 \times 4096 \times 4\text{ bytes} = 16\text{ KiB}$$
1. **L2 Cache Residency:** On an RTX 4070 SUPER, the L2 cache is **36 MiB** (and 72 MiB on an RTX 4090). A 16 KiB buffer *never leaves L2 cache* and never touches DRAM. Therefore, at $M=1$, the separate kernel pipeline incurs **zero DRAM round-trip latency** for the intermediate tensor.
2. **Launch Overhead vs Memory Traffic:** The speedup observed at $M=1$ is strictly due to saving the $3\text{--}5\,\mu\text{s}$ CUDA kernel launch overhead and avoiding the L2 read latency, rather than saving DRAM bandwidth. When evaluating memory-bound fusion claims, one must distinguish between true DRAM bandwidth reduction and launch latency elimination.

---

## 3. Increasing Grid Size Beyond Hardware Saturation in Elementwise Kernels

### The Hypothesis
In `vector_add` and elementwise operations, launching more blocks ensures all SMs stay saturated even in the presence of thread scheduling imbalances.

### The Measurement & Finding
Launching with grid size exceeding $\approx 4\times$ the number of SMs (e.g. $> 224$ blocks on RTX 4070 SUPER, which has 56 SMs) produced flat execution times. Beyond $8\times$ SM capacity, performance slightly degraded due to block scheduling overhead in the GigaThread engine.

### Takeaway
Grid-stride loops (`for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)`) with a fixed grid size of $4\times \text{SM count}$ completely eliminate the need for oversized grid allocations while maintaining 85–91% of theoretical peak bandwidth.

---

## 4. Summary Table of Negative & Neutral Results

| Optimization Tested | Target Kernel | Expected Gain | Observed Result | Primary Bottleneck / Reason |
|---|---|---|---|---|
| Shared-memory bank conflict padding (`[32][33]`) | `matmul_tiled` | 5–10% speedup | -0.2% (neutral/regression) | sm_89 shared-memory crossbar + non-power-of-2 index arithmetic overhead |
| DRAM traffic elimination at $M=1$ | `rmsnorm_linear` | 16 MiB DRAM saved | 0 MiB DRAM saved (16 KiB fits in L2) | 16 KiB intermediate is fully L2 cache resident; speedup is launch overhead, not DRAM |
| Oversized thread grid ($>1024$ blocks) | `vector_add` | Improved tail latency | Flat to +1% latency | 56 SMs fully saturated at 224 blocks; extra blocks increase hardware queue overhead |

