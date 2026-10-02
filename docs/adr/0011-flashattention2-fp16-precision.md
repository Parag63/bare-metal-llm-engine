# ADR 0011 — Native FP16 Precision for FlashAttention-2 with FP32 Softmax Accumulator

**Status:** accepted · **Date:** 2026-10-02 · **Applies to:** Module 4 (`kernels/attention.cu`, `kernels/rope.cu`, `include/engine/kernels.hpp`)

## Context

Module 4 implements FlashAttention-2, Rotary Position Embeddings (RoPE), and Grouped-Query Attention (GQA) for autoregressive transformer inference (TinyLlama-1.1B, LLaMA-2/3).

A key microarchitectural design choice is whether the attention inputs ($Q, K, V$), internal shared-memory tiles, and KV cache should be implemented in **FP32** or **FP16** (`__half` / `engine::half`).

### Constraints and Hardware Realities (NVIDIA Ada Lovelace, sm_89):
1. **Shared-Memory Footprint:**
   - In FlashAttention-2, each thread block computes $O_{\text{tile}} = \text{Softmax}(Q_{\text{tile}} K_{\text{tile}}^T) V_{\text{tile}}$.
   - For a standard tile size of $B_r = 64, B_c = 64$ with head dimension $d = 64$:
     - FP32: $Q, K, V$ tiles consume $(64 \times 64 + 64 \times 64 + 64 \times 64) \times 4\text{ B} = 48\text{ KiB}$ of shared memory per block!
     - On the RTX 4070 SUPER, default shared memory per block is $48\text{ KiB}$ (opt-in up to $99\text{ KiB}$). A 48 KiB allocation limits thread block occupancy to at most **1 active block per SM**, completely starving the SM's warp schedulers.
     - FP16: The identical tile dimensions consume only $24\text{ KiB}$ of shared memory, easily enabling **2–3 active thread blocks per SM** and double-buffered asynchronous pipeline loads (`cp.async`).
2. **KV-Cache Memory Capacity & Bandwidth:**
   - Autoregressive decode is strictly memory-bandwidth bound ($O(N)$ operations for $O(N)$ bytes loaded from the KV cache).
   - In FP16, the KV cache requires 2 bytes per element (e.g., 22 KB per token for TinyLlama-1.1B) rather than 4 bytes in FP32, doubling the effective tokens/s generated per second on a 504 GB/s memory subsystem.
3. **Numerical Precision:**
   - Standard LLaMA and TinyLlama checkpoints store weights and KV activations in FP16 / BF16.
   - Accumulating $QK^T$ and online softmax scaling in FP32 prevents underflow/overflow in $\exp(x - m)$, which provides full numerical fidelity identical to FP32 execution while leveraging FP16 memory efficiency.

## Decision

1. **Native FP16 Inputs & Outputs:**
   - The primary FlashAttention-2 prefill and decode kernel interfaces will accept `const engine::half*` for $Q, K, V$, and output `engine::half* out`.
2. **FP32 Accumulation in Softmax & Tile Reductions:**
   - Dot products $S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$ and online softmax running statistics ($m_i = \max(m_i, s_{ij})$, $l_i = \sum \exp(s_{ij} - m_i)$) will accumulate strictly in **FP32 registers**.
3. **FP16 Shared-Memory Storage:**
   - Shared memory tiles $K_{\text{tile}}$ and $V_{\text{tile}}$ will be stored as `half` (using 16-byte aligned vector loads `uint4`), reducing shared memory occupancy pressure by 50%.
4. **Scalar CPU Reference & Naive GPU Oracle:**
   - The scalar CPU oracle in `src/cpu_ref/` and the naive materialization oracle will support both float and half conversions to validate numerical accuracy within derived error bounds ($rtol = 10^{-3}, atol = 10^{-3}$).

## Consequences

- **Higher SM Occupancy:** Cutting tile memory footprint in half allows multiple concurrent blocks per SM on Ada Lovelace.
- **2× Decode Throughput:** Decode attention streams half as many DRAM bytes from the KV cache.
- **Standardized Pipeline:** Prepares the engine directly for Module 5 (quantization) and Module 6 (paged KV cache), which rely on FP16/BF16 base activations.
