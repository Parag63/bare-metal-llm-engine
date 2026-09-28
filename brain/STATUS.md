# Current Status

> **Last updated:** 2026-09-25
>
> This file is the single source of truth for "what's done, what's in progress, and
> what's next." Update it whenever a milestone is completed.

## Test score (verified on Machine B, RTX 4070 SUPER, sm_89)

```
passed 80   failed 0   pending 0   skipped 0
```
*(CUDA-enabled build with nvcc 12.6, RTX 4070 SUPER — 2026-09-25)*

### Test suite breakdown

| Suite | Tests | Status |
|---|---|---|
| `dtype` | 10 | ✅ All passing |
| `golden` | 6 | ✅ All passing |
| `cpu_ref` | 10 | ✅ All passing |
| `storage` | 4 | ✅ All passing |
| `tensor` | 24 | ✅ All passing (GPU + CPU verified) |
| `kernels` | 26 | ✅ All 26 passing (all 6 kernels + contracts verified) |

## Module completion

### ✅ Infrastructure (100%)

- [x] CMake build system (CUDA-optional, supports WSL & Linux & MSVC)
- [x] Custom test harness (`TEST`, `TEST_PENDING`, `SKIP_TEST`)
- [x] Benchmark harness (CUDA events, warmup, median/min, spread)
- [x] Golden reference generator (`gen_reference.py`)
- [x] Build/test/run scripts (bash + PowerShell)
- [x] `.clang-format`, `.gitignore`
- [x] 5 ADR documents
- [x] Lab notebook template + entries

### ✅ Module 1 — Tensor Library (100% complete & verified)

- [x] `Storage` — refcounted byte buffer, CPU + CUDA allocation, non-owning borrows
- [x] `Tensor::contiguous_strides`, `numel`, `nbytes`, `size`
- [x] `Tensor::to_string`
- [x] Constructor, `Tensor::zeros`
- [x] `Tensor::is_contiguous`
- [x] `Tensor::data`, `Tensor::ptr<T>`
- [x] `Tensor::reshape` (with -1 inference)
- [x] `Tensor::permute`, `Tensor::transpose`
- [x] `Tensor::slice`
- [x] `Tensor::contiguous`, `Tensor::clone`, `Tensor::to`
- [x] `Tensor::from_blob`

### ✅ Module 2 — CUDA Kernel Ladder (6/6 complete & benchmarked)

| # | Kernel | Status | GPU benchmarks? |
|---|---|---|---|
| 1 | `vector_add` | ✅ Complete (worked example) | ✅ Measured (432.6 GB/s, 85.8% peak) |
| 2 | `reduce_sum` | ✅ Complete (two-stage + warp shuffle) | ✅ Measured (458.3 GB/s, 90.9% peak) |
| 3 | `softmax_rows` | ✅ Complete (three-pass + block reduction) | ✅ Measured (435.7 GB/s, 86.4% peak) |
| 4 | `rmsnorm` | ✅ Complete (sum-of-squares + rsqrtf) | ✅ Measured (435.5 GB/s, 86.4% peak) |
| 5 | `matmul_naive` | ✅ Complete (coalesced 16x16 2D mapping) | ✅ Measured (1,852 GFLOP/s @ 4096^3) |
| 6 | `matmul_tiled` | ✅ Complete (shared-memory 32x32 tiled GEMM) | ✅ Measured (2,563 GFLOP/s @ 4096^3, +38% over naive) |

### ❌ Module 3+ — Future Work (not started)

- [ ] Kernel fusion (pre-attention & feedforward fusion)
- [ ] FlashAttention (tiled online softmax + GEMM fusion)
- [ ] Quantization — packed INT4 weights + dequant kernels
- [ ] KV-cache optimization — ring-buffer with zero-copy slicing
- [ ] GGUF-style model loader — mmap'd weights
- [ ] Tokenizer — BPE
- [ ] Sampler — top-k, top-p, temperature
- [ ] Python bindings
- [ ] End-to-end inference: TinyLlama 1.1B
- [ ] Headline result: quantized Llama-2-7B

## Immediate next steps

1. **Kernel Fusion (Module 3)** — combine RMSNorm + QKV projection, or Softmax + Attention
2. **FlashAttention Implementation** — generalise tiled online softmax to avoid materializing $S = QK^T$
3. **Record findings in lab notebook** (`docs/lab-notebook.md`)

## Known issues / blockers

- GPU benchmarks for `reduce_sum` have not been recorded yet (Machine B access needed)
- Lab notebook Week 01 measurements are all "pending"
- `colab/` directory is empty (future use)
