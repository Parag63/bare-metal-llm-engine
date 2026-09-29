# Bare-Metal LLM Inference Engine

A llama.cpp-style LLM inference engine written from scratch in C++17 and CUDA: custom
tensor library, hand-written CUDA kernels, FlashAttention, quantization, KV-cache
optimization, and Python bindings. No PyTorch, no cuBLAS, no ONNX Runtime in the
inference path.

B.Tech CSE major project · Parag Das (2303344) · Jul 2026 – Jun 2027

> **Status: Modules 1 & 2 complete. Moving to kernel fusion and FlashAttention.**
> The tensor library, all six CUDA kernels, and the full verification/benchmark
> infrastructure are implemented and verified. `passed 80  failed 0  pending 0` on
> the CUDA-enabled build (RTX 4070 SUPER, nvcc 12.6, 2026-09-25). Memory-bound
> kernels achieve 85–91% of peak bandwidth; the tiled matmul reaches 2,563 GFLOP/s
> at 4096³ (+38% over naive).

## What "from scratch" means here

Written by hand: the tensor library (shapes, strides, views, reference-counted storage),
every CUDA kernel in the inference path, the attention implementation, the quantization
and dequantization kernels, the KV cache, the model loader, the tokenizer, and the
sampler.

Used as tools, not as implementation: **PyTorch/NumPy** generate float64 reference data
for the test suite, and **cuBLAS** appears in the benchmark binary as a baseline to be
measured against. Neither is linked into the engine
([ADR 0004](docs/adr/0004-cublas-baseline-only.md)).

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

Full setup for both machines, including the Windows + RTX 4090 toolchain and the
clock-locking recipe: **[docs/01-dev-environment.md](docs/01-dev-environment.md)**.

## Layout

```
include/engine/     public headers — dtype, tensor, allocator, device_buffer,
                    check (error handling), cpu_ref, kernels (launch API), cuda_device
src/                C++ implementation (tensor.cpp: 559 lines, Module 1)
src/cpu_ref/        the CPU oracle: scalar, FP64 accumulators, obvious over fast
kernels/            CUDA kernels — all six implemented and benchmarked
tests/              custom harness + suites: dtype, golden, cpu_ref, storage, tensor,
                    kernels (80 tests, all passing)
tests/golden/       generated float64 reference data — gitignored, regenerate it
bench/              bench_cpu_ref (the denominator) and bench_kernels (the results)
tools/              gen_reference.py
cmake/              CUDA detection, warning flags, config header template
brain/              persistent AI/human context — project state, conventions, decisions
runbook/            operational procedures — build, test, benchmark, add-kernel
docs/               environment, exercise ladder, ADRs, lab notebook
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

Six kernels, each introducing one idea and reusing everything before it. All six are
implemented, verified against float64 reference data, and benchmarked on the RTX 4070
SUPER with locked GPU clocks:

| # | Kernel | New idea | GPU result |
|---|---|---|---|
| 1 | `vector_add` | threads, blocks, grid-stride loops | ✅ 432.6 GB/s (85.8% peak) |
| 2 | `reduce_sum` | shared memory, `__syncthreads`, warp shuffles | ✅ 458.3 GB/s (90.9% peak) |
| 3 | `softmax_rows` | per-row reduction, numerical stability | ✅ 435.7 GB/s (86.4% peak) |
| 4 | `rmsnorm` | reusing the reduction pattern | ✅ 435.5 GB/s (86.4% peak) |
| 5 | `matmul_naive` | 2-D indexing, memory traffic problem | ✅ 1,852 GFLOP/s @ 4096³ |
| 6 | `matmul_tiled` | shared-memory tiling and data reuse | ✅ 2,563 GFLOP/s @ 4096³ (+38%) |

Design details in **[docs/02-cuda-exercises.md](docs/02-cuda-exercises.md)**.

Arithmetic intensity, against the RTX 4090's ~82 FLOP/byte balance point, tells you which
resource limits each one: `vector_add` is 0.08 (memory-bound by a factor of a thousand),
`rmsnorm` 0.5, `softmax` 0.6, and matmul at 1024³ is 170 — the first compute-bound
kernel in the project, and the only one where being clever about arithmetic wins anything.

## Design decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-cuda-optional-build.md) | CUDA is detected, never required — the whole project builds and tests without a GPU |
| [0002](docs/adr/0002-storage-tensor-split.md) | `Storage` (owns bytes) and `Tensor` (a view) are separate types, so slices, mmap'd weights and reshapes are free |
| [0003](docs/adr/0003-raw-pointer-kernel-api.md) | Kernel launchers take raw pointers and dimensions, not `Tensor` — decouples the kernel ladder from Module 1 |
| [0004](docs/adr/0004-cublas-baseline-only.md) | cuBLAS is a benchmark baseline, linked into exactly one target, never an implementation |
| [0005](docs/adr/0005-cuda-arch-explicit.md) | `ENGINE_CUDA_ARCH` is explicit (`89`), and mismatches warn loudly — a wrong-arch build otherwise silently JITs and benchmarks nothing meaningful |

## Build options

| Option | Default | Effect |
|---|---|---|
| `ENGINE_WITH_CUDA` | `ON` | Build CUDA kernels if `nvcc` is found. Never fatal when absent |
| `ENGINE_CUDA_ARCH` | `89` | Target architecture. 89 = RTX 4090. A binary built for 89 runs on nothing else |
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
- [x] **Module 2 — CUDA Kernel Ladder** (complete). Six hand-written kernels from
  `vector_add` through `matmul_tiled`, with three-tier verification and GPU benchmarks.
  26 kernel tests, all passing. Memory-bound kernels at 85–91% peak bandwidth.
- [ ] **Kernel Fusion** — combine RMSNorm + QKV projection, or Softmax + Attention, to
  reduce launch overhead (~5 µs/launch × 320 launches/token = ~600 tok/s ceiling).
- [ ] **FlashAttention** — tiled online softmax + GEMM fusion, avoiding materialisation
  of the full S = QK^T attention-score matrix.
- [ ] **Quantization** — packed INT4 weights with dequantisation kernels.
- [ ] **KV-Cache** — ring-buffer cache with zero-copy slicing via the Tensor view system.
- [ ] **Model Loader** — GGUF-style mmap'd weights, ~50 weight tensors aliasing one
  region via non-owning `Storage`.
- [ ] **Tokenizer** — BPE.
- [ ] **Sampler** — top-k, top-p, temperature scaling.
- [ ] **Python Bindings** — for usability.
- [ ] **End-to-End Inference** — TinyLlama 1.1B (dev), quantized Llama-2-7B (headline).

The loader is deliberately architecture-agnostic.
