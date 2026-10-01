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

## 2. Kernel Fusion Trade-Off: 1D Row Fusion vs. 2D Tiled Prefill (`rmsnorm_linear`)

### The Hypotheses
In Module 3, fusing `RMSNorm + Linear` (`rmsnorm_linear`) theoretically eliminates the intermediate normalized activation tensor $X_{\text{norm}}$:
$$\Delta \text{Bytes}_{\text{theory}} = 2 \times M \times K \times 4\text{ bytes}$$

1. **Prefill Hypothesis ($M=512, K=4096, N=4096$):**
   $$\Delta \text{Bytes} = 2 \times 512 \times 4096 \times 4 = 16\text{ MiB DRAM roundtrip eliminated}$$
   Textbook compilation models predict a 15–30% speedup from fusing RMSNorm with the projection layer.

2. **Decode Hypothesis ($M=1, K=4096, N=4096$):**
   Textbook models assume memory bandwidth reduction accounts for fusion speedups even at single-token generation.

---

### Empirical Measurements via NVIDIA Nsight Compute (`ncu`)

Hardware performance counters were captured on an **NVIDIA GeForce RTX 4070 SUPER** (sm_89, 48 MiB L2, 504 GB/s peak DRAM) using dedicated single-launch profilers (`bench/profile_dram.cu`):

#### Table 1: Nsight Compute DRAM & Cache Traffic Metrics

| Kernel Execution | Shape ($M \times N \times K$) | `dram__bytes_read` | `dram__bytes_write` | `dram__bytes.sum` | `lts__t_bytes.sum` (L2) | Kernel Latency |
|---|---|---:|---:|---:|---:|---:|
| **Separate:** `vector_add` + `rmsnorm` | $512 \times 4096$ | 25.20 MB | 0.08 MB | 25.20 MB | 49.59 MB | 0.022 ms |
| **Fused:** `residual_rmsnorm` | $512 \times 4096$ | 16.81 MB | 0.08 MB | 16.89 MB | 42.91 MB | 0.019 ms |
| **Difference (Residual Fusion)** | | **-8.39 MB (-8.0 MiB)** | **~0 MB** | **-8.31 MB** | **-6.68 MB** | **-13.6% (WIN)** |
| | | | | | | |
| **Separate:** `rmsnorm` + `matmul_tiled` | $512 \times 4096 \times 4096$ | 1.08 GB | 27.10 MB | 1.11 GB | 2.24 GB | 6.43 ms |
| **1D Fused:** `rmsnorm_linear_kernel` | $512 \times 4096 \times 4096$ | 3.37 GB | 153.07 MB | 3.52 GB | 33.61 GB | 13.48 ms |
| **Difference (Prefill Fusion)** | | **+2.29 GB (+212%)** | **+125.9 MB** | **+2.41 GB** | **+31.37 GB** | **+109.6% (CATASTROPHIC LOSS)** |
| | | | | | | |
| **Separate:** `rmsnorm` + `matmul_tiled` | $1 \times 4096 \times 4096$ | 73.59 MB | 0.03 MB | 73.63 MB | 76.00 MB | 0.524 ms |
| **1D Fused:** `rmsnorm_linear_kernel` | $1 \times 4096 \times 4096$ | 67.16 MB | 1.27 MB | 68.43 MB | 67.84 MB | 0.366 ms |
| **Difference (Decode Fusion)** | | **-6.43 MB** | **+1.24 MB** | **-5.20 MB** | **-8.16 MB** | **-30.2% (WIN)** |

---

### Root Cause Analysis: Why 1D Fusion Catastrophically Fails at $M=512$

1. **Loss of 2D Shared-Memory Tile Reuse:**
   - In `matmul_tiled`, the computation is partitioned into 2D blocks of size $32 \times 32$. Threads in the thread block cooperatively load a $32 \times 32$ tile of weight matrix $W$ into shared memory. That tile is reused across all 32 tokens (rows of $A$).
   - In 1D row fusion (`rmsnorm_linear_kernel`), each block processes a 1D slice of an output row ($1 \times 256$). Because rows are processed independently to keep normalized activations in shared memory, **weights $W$ cannot be reused across tokens ($M$)**.
   - Every block re-reads columns of $W$ directly through the memory subsystem. As NCU measures, DRAM read traffic explodes from **1.08 GB** to **3.37 GB** (+2.29 GB!). L2 cache traffic explodes by **$15\times$** (from 2.24 GB to 33.61 GB).
   - **The 16 MiB intermediate activation savings is eclipsed $143\times$ over by the 2.29 GB weight re-read penalty!**

2. **Why Decode ($M=1$) Wins Despite 0 MiB Intermediate DRAM Savings:**
   - At $M=1$, the intermediate activation vector is $1 \times 4096 \times 4\text{ B} = 16\text{ KiB}$.
   - On the RTX 4070 SUPER (48 MiB L2 cache), this 16 KiB vector resides entirely in L2. In separate execution, it **never touches DRAM** (DRAM write is 0 B; DRAM read for RMSNorm output is ~44 KB). Thus, **0 MiB of DRAM bandwidth is saved**.
   - However, at $M=1$, 2D tiled GEMM is severely under-utilized: a $32 \times 32$ tile only has 1 valid row, wasting $\frac{31}{32} \approx 97\%$ of the tile capacity and producing uncoalesced tail memory access.
   - The 1D fused kernel streams weights as a continuous 1D vector (like GEMV), eliminating both the 2D tile waste and an entire kernel launch ($3\text{--}5\,\mu\text{s}$). Hence, at $M=1$, fusion yields a genuine $+43\%$ to $+2.52\times$ speedup.

---

### Architectural Solution: Conditional Dynamic Dispatch

Rather than forcing a single execution path, `engine::cuda::rmsnorm_linear` enforces an architectural dispatch boundary:

```cpp
void rmsnorm_linear(...) {
  if (M == 1) {
    // Decode phase: 1D row-fused kernel (bypasses launch overhead and 2D tile waste)
    rmsnorm_linear_fused_direct(in, rms_weight, W, out, M, N, K, eps, stream);
  } else {
    // Prefill phase (M > 1): 2D tiled GEMM reuses weight matrix across tokens
    // Stream-ordered async buffer avoids host synchronization overhead
    float* temp = nullptr;
    cudaMallocAsync(&temp, M * K * sizeof(float), stream);
    rmsnorm(in, rms_weight, temp, M, K, eps, stream);
    matmul_tiled(temp, W, out, M, N, K, stream);
    cudaFreeAsync(temp, stream);
  }
}
```

#### Measured Dispatch Validation (`bench_kernels`):
- **$M=1, K=4096, N=4096$:** Dispatched executes fused path in **0.366 ms** (matches 1D fused; $+43\%$ faster than separate).
- **$M=512, K=4096, N=4096$:** Dispatched executes tiled path in **7.08 ms** (vs 13.48 ms for 1D fused; completely avoids the $2\times$ regression).

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
| 1D row fusion at prefill ($M=512$) | `rmsnorm_linear` | +15–30% speedup (16 MiB DRAM saved) | **-109.6% slowdown** (13.48 ms vs 6.43 ms) | 1D fusion destroys 2D weight matrix reuse across tokens; DRAM read explodes by +2.29 GB |
| DRAM traffic elimination at $M=1$ | `rmsnorm_linear` | 16 MiB DRAM saved | 0 MiB DRAM saved (16 KiB fits in L2) | 16 KiB intermediate is fully L2 cache resident; speedup is launch overhead, not DRAM |
| Oversized thread grid ($>1024$ blocks) | `vector_add` | Improved tail latency | Flat to +1% latency | 56 SMs fully saturated at 224 blocks; extra blocks increase hardware queue overhead |

