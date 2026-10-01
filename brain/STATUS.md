# Current Status

> **Last updated:** 2026-10-01
>
> This file is the single source of truth for "what's done, what's in progress, and
> what's next." Update it whenever a milestone is completed.

## Test score (verified on Machine B, RTX 4070 SUPER, sm_89)

```
passed 107   failed 0   pending 0   skipped 0
```
*(CUDA-enabled build with nvcc 12.6, RTX 4070 SUPER — 2026-10-01)*

### Test suite breakdown

| Suite | Tests | Status |
|---|---|---|
| `dtype` | 10 | ✅ All passing |
| `golden` | 6 | ✅ All passing |
| `cpu_ref` | 23 | ✅ All 23 passing (includes Module 3 fused ops, GEMV, FP16 GEMV, embedding, argmax) |
| `storage` | 4 | ✅ All passing |
| `tensor` | 24 | ✅ All passing (GPU + CPU verified, const and half accessors) |
| `kernels` | 40 | ✅ All 40 passing (12 kernels + contracts verified) |

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
