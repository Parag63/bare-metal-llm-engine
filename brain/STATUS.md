# Current Status

> **Last updated:** 2026-10-01
>
> This file is the single source of truth for "what's done, what's in progress, and
> what's next." Update it whenever a milestone is completed.

## Test score (verified on Machine B, RTX 4070 SUPER, sm_89)

```
passed 118   failed 0   pending 0   skipped 0
```
*(CUDA-enabled build with nvcc 12.6, RTX 4070 SUPER — 2026-10-01)*

### Test suite breakdown

| Suite | Tests | Status |
|---|---|---|
| `dtype` | 10 | ✅ All passing |
| `golden` | 6 | ✅ All passing |
| `cpu_ref` | 25 | ✅ All 25 passing (includes SwiGLU, FP16 GEMV, embedding, argmax) |
| `storage` | 4 | ✅ All passing |
| `tensor` | 24 | ✅ All passing (GPU + CPU verified, const and half accessors) |
| `allocator` | 5 | ✅ All 5 passing (PoolAllocator: alignment, recycling, peak tracking, 100k cycles, device memory) |
| `kernels` | 44 | ✅ All 44 passing (14 kernels + contracts verified, including register-tiled GEMM and SwiGLU) |

---

## Phase 4 Completion: Production Foundation & Register Tiling
- [x] **Flexible Architecture Compilation (ADR 0010):** Authored `docs/adr/0010-multi-architecture-cuda-compilation.md`. Added support for flexible architecture specification in `ENGINE_CUDA_ARCH` with native SASS generation, eliminating runtime JIT overhead.
- [x] **RAII Stream & Event Wrappers:** `include/engine/cuda_stream.hpp`, `src/cuda_stream.cpp` providing move-only zero-overhead wrappers over `cudaStream_t` and `cudaEvent_t` with non-blocking default flags.
- [x] **Pool Allocator & Memory Accounting:** `include/engine/pool_allocator.hpp`, `src/pool_allocator.cpp`. Slab pre-allocation, power-of-two size bucketing (256 B to 1 GiB), 256-byte alignment, current/peak memory tracking. Passed 100,000 cycles acceptance test with `num_driver_allocs == 1` ($< 20$ required).
- [x] **Fused SwiGLU Activation:** `kernels/swiglu.cu`, `src/cpu_ref/swiglu_cpu.cpp`. Single-pass activation reducing DRAM traffic from $20N \to 12N$ bytes. Achieves **0.295 ms** ($1.52\times$ speedup over unfused SiLU+Mul) on $512 \times 11008$ prefill and **0.0051 ms** ($1.61\times$ speedup) on decode. FP16 SwiGLU achieves **0.031 ms** ($9.5\times$ over FP32).
- [x] **2D Register-Tiled GEMM:** `kernels/matmul_register_tiled.cu` ($128 \times 128$ block, $8 \times 8$ register tile, transposed shared memory `s_A`, 128-bit `float4` loads). Increases shared-memory arithmetic intensity by $8\times$ ($0.25 \to 2.0\text{ FLOP/byte}$). Achieves **17.55 TFLOP/s** at $2048^3$ and **16.17 TFLOP/s** at $4096^3$ (**6.7x speedup** over `matmul_tiled` and reaching **73.0% of cuBLAS**).
- [x] **Verification & Artifacts:** 118 / 118 tests passing, lab notebook predictions recorded first, results table and roofline updated.

---

## Phase 3 Completion: FP16 and Missing Inference Kernels
- [x] **FP16 Type Integration:** `include/engine/half.hpp` bridging CUDA `__half` with bit-exact host-side binary16; `Tensor::ptr<half>()` and `const ptr<half>() const`.
- [x] **FP16 GEMV:** `kernels/gemv_fp16.cu` with `half2` vector loads and FP32 accumulator. Achieves **0.075 ms** (vs 0.143 ms FP32, $1.91\times$ speedup) and $1,142.8\text{ GB/s}$ in L2.
- [x] **Token Embedding Gather:** `kernels/embedding.cu` (FP32 & FP16) with 128-bit vector memory instructions (`float4`, `uint4`).
- [x] **Greedy Argmax Sampling:** `kernels/argmax.cu` (FP32 & FP16) with 16-warp shuffle reduction and deterministic tie breaking.
- [x] **Verification & Provenance:** 107 / 107 tests passing, derived error tolerances ($2 \times 10^{-3}$ for FP16), lab notebook predictions recorded first, results table and roofline updated.

---

## Action List Gap Analysis Status (from `plans/llm-engine-action-list.docx`)

### Project-Wide & Infrastructure
- [x] **[Must]** Automated README results table generated via script (`tools/generate_results_table.py`)
- [x] **[Must]** Save environment details with every benchmark run (`driver_version`, `clock_rate_mhz`, `peak_bandwidth_gbs`, `git_hash`, compiler flags)
- [x] **[Must]** Benchmark harness JSON output (`--json` flag on `bench_kernels` and `bench_cpu_ref`)
- [x] **[Should]** CI building CPU-only and running CPU tests (`.github/workflows/ci.yml`)
- [x] **[Should]** Roofline plot script against bandwidth and compute ceilings (`tools/roofline_plot.py` -> `docs/roofline.png`)
- [x] **[Should]** "Negative results" page for non-optimizations and bottlenecks (`docs/negative-results.md`)
- [x] **[Must]** Benchmark script running `llama.cpp`'s `llama-bench` on same machine, model, and prompt lengths (`tools/run_llama_bench.sh`, verified on TinyLlama FP16, Q8_0, Q4_K_M)
- [ ] **[Stretch]** Thin HTTP streaming serving layer (Scheduled for Module 8)

### Completed Modules (1, 2, 3)
- [x] **Module 1 (Tensor Library):** Storage, views, strided slicing, reshape, permute, contiguous, clone, from_blob.
  - *Upcoming refinements:* FP16 end-to-end path (Nov 2026), PoolAllocator + accounting (Nov 2026), Stream/event wrapper (Nov 2026).
- [x] **Module 2 (Kernel Ladder):**
  - [x] 6 ladder kernels: `vector_add`, `reduce_sum`, `softmax_rows`, `rmsnorm`, `matmul_naive`, `matmul_tiled`.
  - [x] **[Must]** GEMV kernel for decode token generation (`kernels/gemv.cu`, achieves **474 GB/s / 94.0% peak BW**, $+14\%$ faster than cuBLAS).
  - *Upcoming refinements:* Register-tiled GEMM + vectorized loads + double buffering to close gap to cuBLAS (Nov 2026).
- [x] **Module 3 (Kernel Fusion):**
  - [x] `rmsnorm_linear` (RMSNorm + Linear projection, eliminates 16 MiB DRAM roundtrip at $M=512$).
  - [x] `residual_rmsnorm` (Residual Add + RMSNorm, eliminates 8 MiB DRAM roundtrip at $M=512$).
  - [x] **[Must]** Tested fusion at decode shapes ($M=1$) vs prefill ($M=512$), documented why 16 KiB intermediate buffer is L2 cache resident in `docs/negative-results.md` and `docs/lab-notebook.md`.
  - [x] **[Should]** Profiling script for Nsight Compute (`scripts/profile_kernels.sh`).

---

## Detailed Roadmap & Updated Timeline (Oct 2026 – Jun 2027)

| Month | Module | Milestone & Deliverables | Priority Items |
|---|---|---|---|
| **Oct 2026** | **Module 4: FlashAttention** | • Naive attention baseline ($S \times S$ matrix)<br>• Causal masking + GQA (32 Q heads, 4 KV heads)<br>• Float64 stability test across sequence lengths<br>• Tiled online softmax kernel + memory crossover benchmark<br>• Separate prefill & decode attention kernels<br>• RoPE, SwiGLU, Embedding lookup, Argmax/sampling kernels | `[Must]`×4<br>`[Should]`×2 |
| **Nov 2026** | **Infrastructure & Ladder Refinements** | • Register-tiled GEMM (`matmul_register_tiled`, float4 loads, double buffer)<br>• End-to-end FP16 Tensor/kernel data path<br>• `PoolAllocator` with memory accounting (peak/current device bytes)<br>• CUDA stream/event asynchronous copy/compute wrapper | `[Must]`×2<br>`[Should]`×2 |
| **Dec 2026 – Jan 2027** | **Module 5: Quantization** | • Q8_0 fallback + Q4_0 real GGUF block layout (32 weights/block + FP16 scale)<br>• Fused dequantize-and-multiply kernel for GEMV (decode) then prefill<br>• Perplexity measurement: FP16 vs Q8 vs Q4 vs `llama.cpp`<br>• Effective bandwidth per token reporting | `[Must]`×3<br>`[Should]`×1 |
| **Feb 2027** | **Module 6: KV-Cache** | • Preallocated per-layer cache + incremental decode correctness test<br>• Memory math in notebook (~22 KB/token for TinyLlama FP16)<br>• Head-major vs sequence-major cache layout comparison<br>• [Stretch] INT8 KV quantization or paged allocator | `[Must]`×2<br>`[Should]`×1<br>`[Stretch]`×1 |
| **Mar 2027** | **Module 7: Model Loader** | • Full GGUF parser (header, metadata, tensor table, alignment)<br>• Memory-mapped weights (`mmap` on Linux/WSL, `MapViewOfFile` on Windows)<br>• Dynamic tokenizer & architecture metadata reading from GGUF<br>• Tensor-by-tensor validation against official `gguf` library<br>• Synthetic GGUF fixture generator script for CI | `[Must]`×3<br>`[Should]`×1 |
| **Apr – May 2027** | **Module 8: End-to-End Inference** | • BPE tokenizer with byte fallback, validated against reference tokenizer<br>• Layer-by-layer comparison tool against PyTorch reference<br>• Greedy, temperature, top-k, top-p sampling<br>• End-to-end TinyLlama 1.1B inference<br>• Headline benchmark: prefill & decode tok/s vs `llama.cpp`<br>• Compare decode speed to memory bandwidth ceiling (~230 tok/s FP16, ~780 tok/s Q4)<br>• [Stretch] CUDA Graphs for decode launch overhead | `[Must]`×5<br>`[Stretch]`×1 |
| **Jun 2027** | **Deliverables & Thesis** | • Gap analysis write-up vs `llama.cpp`<br>• Public lab notebook with predictions vs measurements<br>• Two technical blog posts ("Why tiled GEMM sat at 7% of peak", "Where decode time goes")<br>• Final Major Project Dissertation & Defense | Deliverables×4 |

---

## Immediate Next Steps (October 2026)

1. **Module 4: Naive Attention Baseline** — build $S \times S$ attention matrix reference.
2. **Support Causal Masking & GQA** — 32 Q heads sharing 4 KV heads for TinyLlama.
3. **FlashAttention Implementation** — online softmax tile-by-tile fusion.
4. **Prerequisite Kernels for TinyLlama** — RoPE (Rotary Embeddings) and SwiGLU.
