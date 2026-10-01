# Bare-Metal LLM Inference Engine

A llama.cpp-style LLM inference engine written from scratch in C++17 and CUDA: custom
tensor library, hand-written CUDA kernels, FlashAttention, quantization, KV-cache
optimization, and Python bindings. No PyTorch, no cuBLAS, no ONNX Runtime in the
inference path.

B.Tech CSE major project · Parag Das (2303344) · Jul 2026 – Jun 2027

> **Status: Modules 1, 2 & 3 complete + Phase 4 Foundational Kernels. Moving to FlashAttention (Module 4).**
> The tensor library, 14 hand-written CUDA kernels (including FP16 GEMV, fused SwiGLU, 2D register-tiled GEMM, embedding gather, and argmax), and the full
> verification/benchmark infrastructure are implemented and verified. `passed 118  failed 0  pending 0` across
> 8 test suites on the CUDA-enabled build (RTX 4070 SUPER, nvcc 12.6). Memory-bound
> kernels achieve 85–94% of peak bandwidth (GEMV reaches 473.5 GB/s / 94.0% peak BW, beating cuBLAS by +13.7%);
> 2D register-tiled matmul reaches 16,173 GFLOP/s at 4096³ (73.0% of cuBLAS).

## What "from scratch" means here

Written by hand: the tensor library (shapes, strides, views, reference-counted storage),
every CUDA kernel in the inference path, the attention implementation, the quantization
and dequantization kernels, the KV cache, the model loader, the tokenizer, and the
sampler.

Used as tools, not as implementation: **PyTorch/NumPy** generate float64 reference data
for the test suite, and **cuBLAS** appears in the benchmark binary as a baseline to be
measured against. Neither is linked into the engine
([ADR 0004](docs/adr/0004-cublas-baseline-only.md)).

## Hardware target & development setup

- **Target Hardware (Machine B):** NVIDIA GeForce RTX 4070 SUPER (Ada Lovelace, `sm_89`, 56 SMs, 12 GB GDDR6X, 504.0 GB/s peak bandwidth, locked at 2475 MHz). All empirical measurements reported here were obtained on this device inside WSL2 Ubuntu 24.04 with CUDA 12.6.
- **Host / Laptop (Machine A):** CPU-only development environment running full Tier 1 & Tier 2 test suites with graceful CUDA degradation ([ADR 0001](docs/adr/0001-cuda-optional-build.md)).

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
python3 tools/gen_reference.py           # writes ~28 MiB of float64 reference data
ctest --test-dir build --output-on-failure
build/bin/bench_cpu_ref
```

That works with or without a GPU. With CUDA available you additionally get the `kernels`
test suite and:

```bash
build/bin/bench_kernels
```

Full setup for both machines, including the Windows + RTX 4070 SUPER toolchain and the
clock-locking recipe: **[docs/01-dev-environment.md](docs/01-dev-environment.md)**.

## Layout

```
include/engine/     public headers — dtype, tensor, allocator, device_buffer,
                    check (error handling), cpu_ref, kernels (launch API), cuda_device
src/                C++ implementation (tensor.cpp: 559 lines, Module 1)
src/cpu_ref/        the CPU oracle: scalar, FP64 accumulators, obvious over fast (9 ops)
kernels/            CUDA kernels — all nine implemented (7 ladder + 2 fused ops)
tests/              custom harness + suites: dtype, golden, cpu_ref, storage, tensor,
                    kernels (107 tests, all passing)
tests/golden/       generated float64 reference data — gitignored, regenerate it
bench/              bench_cpu_ref (the denominator) and bench_kernels (with --json support)
tools/              gen_reference.py, generate_results_table.py, roofline_plot.py
cmake/              CUDA detection, warning flags, config header template (bakes git hash)
brain/              persistent AI/human context — project state, conventions, decisions
runbook/            operational procedures — build, test, benchmark, profile-nsight
docs/               environment, ladder, ADRs, lab notebook, roofline.png, negative-results
```

## How correctness is established

Three tiers, and the third is compared against the **first**:

1. `tools/gen_reference.py` computes each operation in float64 (PyTorch, or NumPy with
   `--backend numpy`) and writes binary golden files.
2. `src/cpu_ref/*.cpp` — a scalar C++ oracle — is checked against tier 1.
3. CUDA kernels are checked against **tier 1**, not against tier 2.

If the kernels were checked against the CPU oracle, a misunderstanding shared by both
would pass every test. Comparing each independently against an external float64 answer
means a shared misconception has to survive two implementations and a third-party
numerical reference.

Every tolerance is *derived*, not tuned until green, and each carries its arithmetic in a
comment. `vector_add` is held to bit-exactness; `reduce_sum` is held to a tree-reduction
error bound (`log₂(n)·u·Σ|x|`) so that a kernel which quietly accumulates serially fails
even though its answer looks about right. Details in
[docs/02-cuda-exercises.md](docs/02-cuda-exercises.md).

The harness distinguishes four outcomes, which matters more than it sounds:

| | meaning |
|---|---|
| **pass** | verified against reference data |
| **fail** | broken — exit code 1 |
| **pending** | module not written yet. Not a failure. Reports `[PENDING-PASS]` the moment it starts passing, so it gets promoted |
| **skipped** | prerequisite missing (no GPU, no reference data). Not a failure |

Plus: a stub that throws "not implemented" cannot satisfy a test, and a `--filter` that
matches no test at all exits 2 — so a renamed suite cannot leave behind a green ctest
entry that runs nothing.

## How performance is measured

`bench/bench_harness.hpp` is built around the principle that a number you cannot defend
is worse than no number. It uses CUDA events rather than `std::chrono` (a launch returns
in 3–10 µs, before the kernel runs), discards warmup iterations, reports median and min
rather than mean because GPU timing noise is one-sided, and prints a **spread** column —
`(max − min)/median` — flagging any row above 10% as not trustworthy.

Byte counts use **ideal traffic**: each input read once, each output written once. A naive
matmul's re-reads therefore appear as low achieved bandwidth rather than a fictitious
figure above hardware peak, which is what makes the naive and tiled rows comparable.

Every table carries a provenance footer: device, theoretical peak bandwidth, CUDA arch,
build type, timestamp. Tables go in [docs/lab-notebook.md](docs/lab-notebook.md) with the
locked GPU clock recorded next to them.

`bench_cpu_ref` measures the deliberately-naive CPU reference, and the honest framing of
any speedup is "N× faster than a single-threaded scalar C++ reference implementation" —
not "N× faster than CPU". A blocked multi-threaded AVX2 matmul would be 50–100× closer to
the GPU than that baseline is.

## The kernel ladder (complete)

Fourteen foundational and inference kernels, each introducing one idea and reusing everything before it. All
implemented, verified against float64 reference data, and benchmarked on the RTX 4070
SUPER (504.0 GB/s peak, sm_89) with locked GPU clocks (2475 MHz):

| # | Kernel | New idea | GPU result |
|---|---|---|---|
| 1 | `vector_add` | threads, blocks, grid-stride loops | ✅ 422.8 GB/s (83.9% peak) |
| 2 | `reduce_sum` | shared memory, `__syncthreads`, warp shuffles | ✅ 453.6 GB/s (90.0% peak) |
| 3 | `softmax_rows` | per-row reduction, numerical stability | ✅ 434.0 GB/s (86.1% peak) |
| 4 | `rmsnorm` | reusing the reduction pattern | ✅ 433.0 GB/s (85.9% peak) |
| 5 | `matmul_naive` | 2-D indexing, memory traffic problem | ✅ 1738 GFLOP/s @ 4096^3 |
| 6 | `matmul_tiled` | shared-memory tiling and data reuse | ✅ 2407 GFLOP/s @ 4096^3 (+39%) |
| 7 | `gemv` | decode token projection (M=1), 128-bit vector loads | ✅ 461.8 GB/s (91.6% peak) |
| 8 | `residual_rmsnorm` | fused elementwise add + row reduction in 1 pass | ✅ 0.033 ms (+11.1% over separate) |
| 9 | `rmsnorm_linear` | fused activation normalization + linear projection | ✅ 0.609 ms (+47.0% decode M=1) |
| 10 | `gemv_fp16` | decode token projection in FP16 with FP32 accumulator | ✅ 0.076 ms (1.89× speedup over FP32) |
| 11 | `embedding` | token gather from row-major embedding table (FP32 & FP16) | ✅ 0.009 ms (128-bit vector loads) |
| 12 | `argmax` | greedy token sampling via 16-warp shuffle reduction | ✅ 0.010 ms (deterministic tie-breaking) |
| 13 | `matmul_register_tiled` | 2D register tiling (8x8 thread tile, outer products, float4) | ✅ 16173 GFLOP/s @ 4096^3 (6.7× over tiled) |
| 14 | `swiglu` | fused SiLU + elementwise multiply (40% memory traffic reduction) | ✅ 0.295 ms (1.52× over unfused SiLU+Mul) |

### Kernel fusion (Module 3 — complete)

Fusing memory-bound operations between transformer sub-layers to eliminate DRAM round-trips:

| # | Kernel | Fused operations | Design rationale |
|---|---|---|---|
| 8 | `rmsnorm_linear` | RMSNorm + Linear projection | Keeps normalized row in shared memory; saves 16 MiB DRAM round-trip per layer at M=512 |
| 9 | `residual_rmsnorm` | Residual Add + RMSNorm | Computes residual sum and normalized state in a single pass; saves 8 MiB DRAM traffic |

Design details in **[docs/02-cuda-exercises.md](docs/02-cuda-exercises.md)** and **[ADR 0006](docs/adr/0006-kernel-fusion-strategy.md)**. Complete reference diagrams in **[docs/llm-inference-flowchart.md](docs/llm-inference-flowchart.md)**.
Empirical analysis of non-optimizations and bottlenecks is recorded in **[docs/negative-results.md](docs/negative-results.md)**, and full hardware roofline curves are visualized in **[docs/roofline.png](docs/roofline.png)**.

Arithmetic intensity, against the RTX 4070 SUPER's ~70 FLOP/byte balance point, tells you which
resource limits each one: `vector_add` is 0.08 (memory-bound by a factor of nearly a thousand),
`gemv` is 0.25–0.50, `rmsnorm` 0.5, `softmax` 0.6, and matmul at 4096³ is ~170 — the first compute-bound
kernel in the project, and the only one where being clever about arithmetic wins anything.

### Benchmark Credibility & Baseline Verification

| Baseline Comparison | Configuration | Baseline Result | Engine Result | Ratio / Notes |
|---|---|---:|---:|---|
| **cuBLAS SGEMM vs Tiled GEMM** | 4096³ FP32 | 22148 GFLOP/s (cuBLAS) | 2407 GFLOP/s (tiled) | 10.9% of cuBLAS (hand-written FP32 SIMT vs Tensor Cores) |
| **Register-Tiled GEMM vs cuBLAS** | 4096³ FP32 | 22148 GFLOP/s (cuBLAS) | 16173 GFLOP/s (register-tiled) | 73.0% of cuBLAS (6.7× over tiled) |
| **GEMV Cold DRAM vs Warm L2** | 1x4096x4096 (decode) | 461.8 GB/s (warm L2) | 461.8 GB/s (cold DRAM) | 91.6% peak DRAM (pure streaming via rotating weight buffers) |
| **Fused vs Unfused SwiGLU** | 512x11008 (prefill MLP) | 0.448 ms (unfused) | 0.295 ms (fused) | 1.52× speedup (12N vs 20N bytes DRAM traffic) |
| **llama-bench External Baseline** | TinyLlama-1.1B (Q4_K_M) | 18,512.0 t/s (pp512) | 391.2 t/s (tg128) | 4-bit quantized weights (~249 GB/s effective) |
| **llama-bench External Baseline** | TinyLlama-1.1B (Q8_0) | 18,767.1 t/s (pp512) | 275.2 t/s (tg128) | 8-bit quantized weights (~300 GB/s effective) |
| **llama-bench External Baseline** | TinyLlama-1.1B (FP16) | 21,357.9 t/s (pp512) | 181.3 t/s (tg128) | 16-bit unquantized weights (~372 GB/s effective) |

## Design decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-cuda-optional-build.md) | CUDA is detected, never required — the whole project builds and tests without a GPU |
| [0002](docs/adr/0002-storage-tensor-split.md) | `Storage` (owns bytes) and `Tensor` (a view) are separate types, so slices, mmap'd weights and reshapes are free |
| [0003](docs/adr/0003-raw-pointer-kernel-api.md) | Kernel launchers take raw pointers and dimensions, not `Tensor` — decouples the kernel ladder from Module 1 |
| [0004](docs/adr/0004-cublas-baseline-only.md) | cuBLAS is a benchmark baseline, linked into exactly one target, never an implementation |
| [0005](docs/adr/0005-cuda-arch-explicit.md) | `ENGINE_CUDA_ARCH` is explicit (`89`), and mismatches warn loudly — a wrong-arch build otherwise silently JITs and benchmarks nothing meaningful |
| [0006](docs/adr/0006-kernel-fusion-strategy.md) | Kernel fusion strategy — fuse memory-bound elementwise operations between sub-layers to eliminate DRAM traffic |
| [0007](docs/adr/0007-gemv-decode-specialization.md) | Dedicated GEMV kernel — 1D column tiling and warp reduction over $K$ eliminates 2D GEMM idle threads at $M=1$ |
| [0008](docs/adr/0008-benchmark-harness-provenance.md) | Benchmark harness JSON logging & git provenance — structured JSON output with baked git hash, driver, and clocks |
| [0009](docs/adr/0009-negative-results-reporting.md) | Negative results reporting — empirical documentation of non-optimizations and microarchitectural boundaries |
| [0010](docs/adr/0010-multi-architecture-cuda-compilation.md) | Multi-architecture CUDA compilation — flexible architecture lists in `ENGINE_CUDA_ARCH` with native SASS generation |

## Build options

| Option | Default | Effect |
|---|---|---|
| `ENGINE_WITH_CUDA` | `ON` | Build CUDA kernels if `nvcc` is found. Never fatal when absent |
| `ENGINE_CUDA_ARCH` | `89` | Target architecture. 89 = RTX 4070 SUPER (Ada Lovelace). A binary built for 89 runs on nothing else |
| `ENGINE_BUILD_TESTS` | `ON` | The `engine_tests` binary and its ctest entries |
| `ENGINE_BUILD_BENCH` | `ON` | `bench_cpu_ref`, and `bench_kernels` when CUDA is on |
| `ENGINE_BENCH_CUBLAS` | `ON` | Measure `cublasSgemm` as a matmul baseline. Degrades to a warning if cuBLAS is absent |
| `ENGINE_SYNC_CHECK_KERNELS` | `OFF` | Synchronise after every launch, so faults name the launch that caused them. **Invalidates all benchmarks** — use a separate build directory |
| `ENGINE_WERROR` | `OFF` | Warnings as errors |

Useful targets: `cmake --build build --target reference_data` regenerates the golden
files; `--target bench` runs the benchmark binaries.

## Roadmap

- [x] **Module 1 — Tensor Library** (complete). `Storage` + `Tensor` with refcounted
  views, reshape, transpose, slice, contiguous, clone, device transfer. 24 tensor tests,
  4 storage tests, all passing.
- [x] **Module 2 — CUDA Kernel Ladder & GEMV** (complete). Seven hand-written kernels from
  `vector_add` through `gemv` (decode specialization), with three-tier verification and GPU benchmarks.
  35 kernel tests, all passing. Memory-bound kernels reach up to 94.0% peak bandwidth.
- [x] **Module 3 — Kernel Fusion** (complete). Fused RMSNorm + Linear projection (exercise 8)
  and Fused Residual Add + RMSNorm (exercise 9). Eliminates intermediate DRAM roundtrips
  and kernel launch overhead. Verified against golden reference data.
- [x] **Phase 4 Deliverables — Foundational Kernels & Infrastructure** (complete). FP16 GEMV, Fused SwiGLU (>80% peak BW),
  2D Register-Tiled GEMM (16.2 TFLOP/s), Embedding Lookup, Argmax Greedy Sampling, Stream-ordered Pool Allocator,
  RAII CUDA Streams & Events, and Multi-Arch Compilation (ADR 0010). 118 / 118 unit tests passing across 8 suites.
- [ ] **Module 4 — FlashAttention-2** — tiled online softmax + GEMM fusion, avoiding materialisation
  of the full S = QK^T attention-score matrix; causal masking + GQA support; RoPE and SwiGLU.
- [ ] **Module 5 — Quantization** — packed INT4 / INT8 weights with fused dequantisation GEMV kernels.
- [ ] **Module 6 — KV-Cache** — ring-buffer cache with zero-copy slicing via the Tensor view system.
- [ ] **Module 7 — Model Loader** — GGUF-style mmap'd weights, ~50 weight tensors aliasing one
  region via non-owning `Storage`.
- [ ] **Module 8 — End-to-End Inference & Evaluation** — BPE Tokenizer, Sampler, TinyLlama 1.1B execution loop,
  Python bindings (`pybind11`), and `llama-bench` comparative evaluation.

The loader is deliberately architecture-agnostic.
