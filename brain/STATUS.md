# Current Status

> **Last updated:** 2026-11-11
>
> This file is the single source of truth for "what's done, what's in progress, and
> what's next." Update it whenever a milestone is completed.

## Test score (verified on Machine B, RTX 4070 SUPER, sm_89)

```
passed 141   failed 0   pending 0   skipped 0
```
*(CUDA-enabled build with nvcc 12.6, RTX 4070 SUPER — 2026-11-11, from `./build/bin/engine_tests --list`)*

### Test suite breakdown

| Suite | Tests | Status |
|---|---:|---|
| `dtype` | 10 | ✅ All passing |
| `golden` | 6 | ✅ All passing |
| `cpu_ref` | 30 | ✅ All 30 passing (includes SwiGLU, FP16 GEMV, embedding, argmax, RoPE, attention, attention_online) |
| `storage` | 3 | ✅ All passing |
| `tensor` | 35 | ✅ All passing (GPU + CPU verified, const and half accessors) |
| `allocator` | 8 | ✅ All 8 passing (PoolAllocator: alignment, recycling, peak tracking, 100k cycles, multi-stream events) |
| `kernels` | 49 | ✅ All 49 passing (18 kernels: ladder, fusion, RoPE, attention naive/prefill/decode, bridge/subblock integration) |
| **Total (`all`)** | **141** | **✅ All passing across 8 CTest entries (7 suites + `all`)** |

---

## Module 4 Completion: Attention, RoPE, FlashAttention-2 & FlashDecoding
- [x] **ADRs & Contracts (`ADR-0011`, `ADR-0012`, `ADR-0013`):** Native FP16 I/O with FP32 accumulation/online-softmax; token-major `Q`/`Out` `[S, H_q, D]`, head-major strided `K`/`V` `[H_kv, S_max, D]`, FP32 `cos`/`sin` tables.
- [x] **CPU Reference Oracles & Golden Reference Suite:** `cpu::rope_precompute_tables`, `cpu::rope`, `cpu::attention`, and tiled `cpu::attention_online` verified across 99 binary reference files (66.8 MiB).
- [x] **Rotary Position Embeddings (`kernels/rope.cu`):** Fused single-launch Q+K RoPE supporting `RotateHalf` & `AdjacentPair`; **0.044 ms (439.1 GB/s, 87.1% peak BW)** at prefill `S=2048`.
- [x] **Naive Baseline & FlashAttention-2 Prefill (`kernels/attention_naive.cu`, `kernels/attention_prefill.cu`):** Eliminated `[H_q, S, S]` score matrix (**115× DRAM reduction** before softmax via `ncu`: `10.56 MB` vs `>1.16 GiB`), reaching **14.02 TFLOP/s** causal at `S=4096` (**4.89 ms**, **16.1×** faster than naive `78.61 ms`, **+20.9%** from GQA SMEM group sharing).
- [x] **FlashDecoding Split-KV & Warp-Per-Head GQA Broadcast (`kernels/attention_decode.cu`):** Solved `kGChunk=8` 138-reg collapse (`53 regs`, `66.7%` occ, 0 spills, 0 bank conflicts); reaches **463.6 GB/s (91.97% peak BW)** on Llama-2-7B `L=8192` and **0.088 ms (381.7 GB/s, 17.8× vs single-pass)** on TinyLlama `L=32768`.
- [x] **Sub-Block Integration Bridge (`kernels/attention_bridge.cu`):** Verified prefill + multi-step decode KV cache append at exact TinyLlama `d_model=2048` dimensions.
- [x] **Nsight Compute, Roofline & `PoolAllocator` Fix (`src/pool_allocator.cpp`):** Confirmed prefill math-bound (`62–84%` math throttle) and decode memory-bound (`1.005×` cold DRAM bytes); capped auto-slab growth at `64 MiB` + reverse slab scan to prevent `4 GiB` WDDM spill after naive `S=4096`.
- [x] **Decode `rmsnorm_linear` Baseline Reality:** Exposed strawman baseline; fair baseline `rmsnorm + gemv` achieves **0.149 ms (465 GB/s)**, beating 1D strip fusion (0.366 ms) by **$2.45\times$**. Updated dispatch and recorded in `docs/negative-results.md`.
- [x] **`swiglu` Accounting Discrepancy Resolved:** Fused FP32 SwiGLU achieves **0.286 ms** (moving 67.6 MB at **236.4 GB/s / 46.9% peak BW**, yielding a **$1.55\times$ speedup** over unfused 0.444 ms). Erroneous 410 GB/s accounting claim removed.
- [x] **`residual_rmsnorm` L2 Residency Disclosed:** $512 \times 4096$ working set (33.55 MB) resides in 48 MB L2; true cold DRAM streaming measured at $4096 \times 4096$ (268 MB) achieving **436.3 GB/s (86.56% peak BW)** in **0.615 ms**.
- [x] **Prefill `rmsnorm_linear` GFLOP/s Corrected:** Updated from 16,170 to **17,353 GFLOP/s** for $512 \times 4096 \times 4096$ in 0.99 ms.
- [x] **Non-Tile Multiple Tests for `matmul_register_tiled`:** Added `TEST(kernels, matmul_register_tiled_non_tile_multiples)` in `tests/test_kernels.cu`, testing $\{65 \times 137 \times 73\}$, $\{3 \times 7 \times 11\}$, $\{129 \times 257 \times 65\}$, $\{1 \times 65 \times 127\}$, and $\{71 \times 97 \times 113\}$ with 100% pass rate.
- [x] **CI Verification Confirmed:** Verified that all 3 failing checks (`clang-format`, `linux-cuda-compile`, `windows-msvc`) are green in GitHub Actions run `36911092275`.
- [x] **Attention Precision Architectural Decision (ADR-0011):** Formally decided to start Module 4 FlashAttention-2 natively in **FP16** with FP32 online softmax accumulator to minimize shared memory consumption.

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
| **Oct 2026** | **Module 4: FlashAttention-2** | • Rotary Position Embeddings (RoPE) kernel (`kernels/rope.cu`)<br>• Naive attention baseline with causal masking & GQA ($S \times S$ reference)<br>• Float64 stability test across sequence lengths ($S \in [64, 2048]$)<br>• Tiled online softmax FlashAttention-2 kernel + memory crossover benchmark<br>• Separate prefill & decode (FlashDecoding) attention kernels | `[Must]`×4<br>`[Should]`×2 |
| **Nov 2026** | **Infrastructure & Ladder Refinements** | • Register-tiled GEMM (`matmul_register_tiled`, float4 loads, double buffer)<br>• End-to-end FP16 Tensor/kernel data path<br>• `PoolAllocator` with memory accounting (peak/current device bytes)<br>• CUDA stream/event asynchronous copy/compute wrapper | `[Must]`×2<br>`[Should]`×2 |
| **Dec 2026 – Jan 2027** | **Module 5: Quantization** | • Q8_0 fallback + Q4_0 real GGUF block layout (32 weights/block + FP16 scale)<br>• Fused dequantize-and-multiply kernel for GEMV (decode) then prefill<br>• Perplexity measurement: FP16 vs Q8 vs Q4 vs `llama.cpp`<br>• Effective bandwidth per token reporting | `[Must]`×3<br>`[Should]`×1 |
| **Feb 2027** | **Module 6: KV-Cache** | • Preallocated per-layer cache + incremental decode correctness test<br>• Memory math in notebook (~22 KB/token for TinyLlama FP16)<br>• Head-major vs sequence-major cache layout comparison<br>• [Stretch] INT8 KV quantization or paged allocator | `[Must]`×2<br>`[Should]`×1<br>`[Stretch]`×1 |
| **Mar 2027** | **Module 7: Model Loader** | • Full GGUF parser (header, metadata, tensor table, alignment)<br>• Memory-mapped weights (`mmap` on Linux/WSL, `MapViewOfFile` on Windows)<br>• Dynamic tokenizer & architecture metadata reading from GGUF<br>• Tensor-by-tensor validation against official `gguf` library<br>• Synthetic GGUF fixture generator script for CI | `[Must]`×3<br>`[Should]`×1 |
| **Apr – May 2027** | **Module 8: End-to-End Inference** | • BPE tokenizer with byte fallback, validated against reference tokenizer<br>• Layer-by-layer comparison tool against PyTorch reference<br>• Greedy, temperature, top-k, top-p sampling<br>• End-to-end TinyLlama 1.1B inference<br>• Headline benchmark: prefill & decode tok/s vs `llama.cpp`<br>• Compare decode speed to memory bandwidth ceiling (~230 tok/s FP16, ~780 tok/s Q4)<br>• [Stretch] CUDA Graphs for decode launch overhead | `[Must]`×5<br>`[Stretch]`×1 |
| **Jun 2027** | **Deliverables & Thesis** | • Gap analysis write-up vs `llama.cpp`<br>• Public lab notebook with predictions vs measurements<br>• Two technical blog posts ("Why tiled GEMM sat at 7% of peak", "Where decode time goes")<br>• Final Major Project Dissertation & Defense | Deliverables×4 |

---

## Immediate Next Steps (Module 5: Quantization)

1. **Module 5.1 (`Q8_0` & `Q4_0` Layouts):** Implement block quantization formats matching real GGUF block layouts (32 weights/block + FP16 scale) and host float64 references.
2. **Module 5.2 (Fused Dequantize+GEMV / GEMM):** Implement decode dequant+GEMV and prefill dequant+GEMM kernels with effective bandwidth and perplexity verification against `llama.cpp`.
