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

## 2. Kernel Fusion Reality: 1D Strip Fusion Defeated at Both Prefill AND Decode (`rmsnorm_linear`)

### The Hypotheses
In Module 3, fusing `RMSNorm + Linear` (`rmsnorm_linear`) into a single 1D strip kernel was intended to eliminate intermediate normalized activation memory traffic:
$$\Delta \text{Bytes}_{\text{theory}} = 2 \times M \times K \times 4\text{ bytes}$$

1. **Prefill Hypothesis ($M=512, K=4096, N=4096$):**
   Eliminating the 16 MiB intermediate activation roundtrip was hypothesized to yield a 15–30% speedup.
2. **Decode Hypothesis ($M=1, K=4096, N=4096$):**
   1D row fusion was originally claimed to provide a "+43% win" over separate execution.

---

### The Audit: Exposing the Strawman Baseline at $M=1$

The initial "+43% win" claimed for decode fusion compared 1D fused `rmsnorm_linear_kernel` (0.366 ms / 183 GB/s) against **`rmsnorm + matmul_tiled` at $M=1$** (0.524 ms / 64 GB/s).

This was a **weak strawman baseline**:
`matmul_tiled` is a 2D tile kernel ($32 \times 32$). At $M=1$, 31 of 32 rows are dummy padding, causing massive thread divergence and wasting ~97% of compute capacity.

The **fair, production baseline** for single-token decode is `rmsnorm` followed by specialized `gemv`:
- `rmsnorm(1, 4096)`: **0.005 ms** (streams $1 \times 4096 \times 4\text{ B} = 16\text{ KiB}$)
- `gemv(4096, 4096)`: **0.144 ms** (streams 64.03 MiB at **465.4 GB/s**, 92.3% peak BW)
- **Fair Separate Baseline (`rmsnorm + gemv`):** $0.005 + 0.144 =$ **0.149 ms**!

Against this fair baseline:
- 1D Fused `rmsnorm_linear`: **0.366 ms** (182.0 GB/s)
- Fair Baseline (`rmsnorm + gemv`): **0.149–0.153 ms** (439–465 GB/s)
- **Result:** The 1D fused kernel is **$2.45\times$ SLOWER** than separate execution!

---

### Table 1: Comprehensive Prefill and Decode Comparison (RTX 4070 SUPER)

| Kernel Execution | Shape ($M \times N \times K$) | Measured Latency | Achieved Bandwidth | % Peak BW | Outcome vs Fair Baseline |
|---|---|---:|---:|---:|---|
| **1D Fused:** `rmsnorm_linear_kernel` | $1 \times 4096 \times 4096$ (decode) | 0.366–0.369 ms | 182.0 GB/s | 36.1% | **$2.45\times$ SLOWER (DEFEAT)** |
| **Weak Baseline:** `rmsnorm + matmul_tiled` | $1 \times 4096 \times 4096$ (decode) | 0.524–0.534 ms | 62.9 GB/s | 12.5% | Artificial strawman |
| **Fair Baseline:** `rmsnorm + gemv` | $1 \times 4096 \times 4096$ (decode) | **0.149–0.153 ms** | **439.0–465.4 GB/s** | **87.1–92.3%** | **WINNER (Fastest)** |
| | | | | | |
| **1D Fused:** `rmsnorm_linear_kernel` | $512 \times 4096 \times 4096$ (prefill) | 13.48–14.17 ms | 5.92 GB/s | 1.17% | **$13.6\times$ SLOWER (CATASTROPHIC)** |
| **Separate:** `rmsnorm + matmul_tiled` | $512 \times 4096 \times 4096$ (prefill) | 6.43–7.18 ms | 12.85 GB/s | 2.55% | Tiled 2D baseline |
| **Separate:** `rmsnorm + matmul_register_tiled` | $512 \times 4096 \times 4096$ (prefill) | **0.99–1.42 ms** | — | — | **WINNER (17.35 TFLOP/s, 73% cuBLAS)** |

---

### Why 1D Strip Fusion Fails in BOTH Regimes

1. **Why It Fails at Decode ($M=1$): Severe Under-Parallelization & Scalar Reductions**
   - With $N = 4096$ and block size 256, the 1D fused kernel launches only $\lceil 4096 / 256 \rceil = 16$ blocks.
   - The RTX 4070 SUPER has **56 SMs**. Launching 16 blocks leaves **40 SMs (71% of the GPU) completely idle!**
   - Inside each block, a single thread loops over $K=4096$ elements with unvectorized scalar loads.
   - In contrast, `gemv` launches **64 blocks of 8 warps** (256 threads), saturating all 56 SMs. Each block computes 64 columns using `float2` 64-bit vector loads, and 8 warps cooperatively reduce over $K$ in shared memory.

2. **Why It Fails at Prefill ($M=512$): Catastrophic Weight Reload Penalty**
   - In 1D row fusion, each block processes one row independently to hold normalized activations in shared memory. Consequently, **weights $W$ cannot be reused across tokens**.
   - As verified by Nsight Compute (`ncu`), DRAM read traffic explodes from **0.82 GB to 3.37 GB (+2.55 GB!)**.
   - The minor 16 MiB intermediate activation savings is obliterated $160\times$ over by the 2.55 GB weight reload penalty.

---

### Architectural Decision & Production Dispatch

1D strip fusion is a **complete negative result** across all operational dimensions.
The engine's `rmsnorm_linear` dispatches as follows:
- **$M = 1$ (Decode):** Dispatches to `rmsnorm` followed by `gemv` (achieving ~0.15 ms / 465 GB/s).
- **$M > 1$ (Prefill):** Dispatches to `rmsnorm` followed by `matmul_register_tiled` (achieving ~1.0 ms / 17.35 TFLOP/s).
- **Future Alternative:** Load-time weight folding (pre-scaling $W$ by $1/\text{rms}$ with per-row epilogue scale) or true 2D fused register-tiled GEMM.

---

## 3. Cache Residency vs. Cold DRAM Streaming (`residual_rmsnorm` & `swiglu`)

### 1. `residual_rmsnorm` L2 Residency at $512 \times 4096$
- At shape $512 \times 4096$, the kernel reads 2 input tensors and writes 2 output tensors.
- Each tensor is $512 \times 4096 \times 4\text{ B} = 8.39\text{ MB}$.
- Total working set is $4 \times 8.39\text{ MB} = \mathbf{33.55\text{ MB}}$.
- The RTX 4070 SUPER has a **48 MiB L2 cache**. Since $33.55\text{ MB} < 48\text{ MB}$, the working set resides entirely in L2 across repeated benchmark iterations.
- An execution time of $19\text{--}34\,\mu\text{s}$ implies a throughput of $33.55\text{ MB} / 34\,\mu\text{s} \approx \mathbf{990\text{--}1,765\text{ GB/s}}$ (which exceeds the 504 GB/s theoretical peak of GDDR6X DRAM by $2\text{--}3.5\times$).
- **Cold DRAM Reality:** Benchmarking at $4096 \times 4096$ ($4 \times 67.1\text{ MB} = \mathbf{268.4\text{ MB}}$, which overflows the 48 MB L2 cache) yields a median of **0.615 ms**, corresponding to an honest cold DRAM streaming bandwidth of **436.3 GB/s (86.56% of peak DRAM BW)**.

### 2. `swiglu` Accounting vs. Latency Reality
- At $512 \times 11008$, total elements $N = 5,636,096$.
- Memory traffic at 12 bytes/elem (gate read 4B, up read 4B, out write 4B) is strictly:
  $$\text{Traffic} = 3 \times 5,636,096 \times 4\text{ B} = \mathbf{67.63\text{ MB}}$$
- An audit table claim of $410.2\text{ GB/s}$ alongside a median latency of $0.295\text{ ms}$ was an accounting discrepancy ($67.63\text{ MB} / 0.295\text{ ms} = 229.3\text{ GB/s}$). A throughput of 410 GB/s would mathematically require a latency of $\le 0.165\text{ ms}$.
- **Empirical Measurement:**
  - Unfused (SiLU + Mul): **0.444 ms** (moving $20N = 112.7\text{ MB}$ at 253.7 GB/s)
  - Fused FP32: **0.286 ms** (moving $12N = 67.6\text{ MB}$ at **236.4 GB/s / 46.9% peak BW**)
  - Speedup is a genuine **$1.55\times$**, directly tracking the theoretical $1.67\times$ byte reduction.
  - FP16 SwiGLU (`swiglu_fp16`) moves $6N = 33.8\text{ MB}$ in **0.030 ms** ($9.5\times$ over FP32), fitting partially inside L2 cache.

---

## 4. Increasing Grid Size Beyond Hardware Saturation in Elementwise Kernels

### The Hypothesis
In `vector_add` and elementwise operations, launching more blocks was thought to improve SM occupancy.

### The Finding
Launching grids exceeding $4\times$ SM count ($> 224$ blocks on RTX 4070 SUPER) yields flat performance, and beyond $8\times$ degrades slightly due to block scheduling queue contention. Grid-stride loops with $4\times \text{SM count}$ achieve optimal saturation.

---

## 5. Summary Table of Negative & Neutral Results

| Optimization Tested | Target Kernel | Expected Gain | Observed Result | Primary Bottleneck / Reason |
|---|---|---|---|---|
| 1D Strip Fusion at Decode ($M=1$) | `rmsnorm_linear` | +43% vs separate | **$2.45\times$ SLOWER** vs `rmsnorm + gemv` | 16 blocks leaves 40 of 56 SMs idle; unvectorized scalar reduction vs multi-warp `gemv` |
| 1D Row Fusion at Prefill ($M=512$) | `rmsnorm_linear` | +15–30% speedup (16 MiB saved) | **$13.6\times$ SLOWER** vs register-tiled | Destroys 2D weight reuse across tokens; DRAM read explodes by +2.55 GB |
| Shared-memory bank conflict padding (`[32][33]`) | `matmul_tiled` | 5–10% speedup | -0.2% (neutral/regression) | sm_89 shared-memory crossbar + non-power-of-2 index arithmetic overhead |
| DRAM traffic claim for warm L2 residual | `residual_rmsnorm` | 442 GB/s cold DRAM | 33.5 MB resides in 48 MB L2 | 19–34 µs is an L2 cache hit measurement; true cold DRAM is 436 GB/s at 4096×4096 |
| Oversized thread grid ($>1024$ blocks) | `vector_add` | Improved tail latency | Flat to +1% latency | 56 SMs fully saturated at 224 blocks; extra blocks increase hardware queue overhead |


