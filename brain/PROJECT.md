# Project Identity

> Read this file first. It tells any AI agent or new contributor exactly what this
> project is, how to talk about it, and what they must never do.

## What this is

**Bare-Metal LLM Inference Engine** — a llama.cpp-style LLM inference engine written
from scratch in C++17 and CUDA. Custom tensor library, hand-written CUDA kernels,
FlashAttention, quantization, KV-cache optimization, and Python bindings.

**No PyTorch, no cuBLAS, no ONNX Runtime in the inference path.** These tools are used
only for *verification* (PyTorch/NumPy generate reference data) and *benchmarking*
(cuBLAS appears as a baseline to measure against).

## Project metadata

| Field | Value |
|---|---|
| Author | Parag Das (2303344) |
| Type | B.Tech CSE Major Project |
| Timeline | Jul 2026 – Jun 2027 |
| Current Status | Modules 1–3 + Phase 4 Complete (Oct 2026) · 118/118 Tests Passing |
| Language | C++17 (engine), CUDA (kernels), Python (tools) |
| Build | CMake ≥ 3.20 |
| Target GPUs | NVIDIA GeForce RTX 4070 SUPER (sm_89, 56 SMs, 504 GB/s) · Reference: RTX 4090 (sm_89) |
| Dev model | TinyLlama 1.1B |
| Headline model | Quantized Llama-2-7B |

## Project Timeline & Milestones

| Phase / Module | Timeframe | Focus | Status |
|---|---|---|---|
| **Module 1: Tensor Library** | Jul – Aug 2026 | `Storage`/`Tensor` split, zero-copy slicing, shape/strides | ✅ Complete (559 lines, 100% tests) |
| **Module 2: CUDA Kernel Ladder** | Aug – Sep 2026 | Kernels 1–6 (`vector_add` through `matmul_tiled`) | ✅ Complete (Up to 90.9% peak BW) |
| **Module 3: Kernel Fusion & GEMV** | Sep – Oct 2026 | Kernels 7–9 (`gemv`, `rmsnorm_linear`, `residual_rmsnorm`) | ✅ Complete (94.0% peak BW, cuBLAS beaten on M=1) |
| **Phase 4: Production & Ladder Refinement** | Oct 2026 | Register-tiled GEMM (73% of cuBLAS), PoolAllocator, Fused SwiGLU, sm_89 native tuning | ✅ Complete (118/118 tests, 17.5 TFLOP/s) |
| **Module 4: FlashAttention-2** | Oct – Nov 2026 | Tiled online softmax, shared memory PV, GQA support | 🔄 Next Up (Oct 2026) |
| **Module 5: Weight Quantization** | Nov – Dec 2026 | Packed INT4 / INT8 dequantization, GEMV AWQ/GPTQ kernels | 📅 Scheduled |
| **Module 6: KV-Cache Optimization** | Jan – Feb 2027 | PagedAttention / ring-buffer zero-copy slicing, RoPE | 📅 Scheduled |
| **Module 7: GGUF Model Pipeline** | Mar – Apr 2027 | mmap loader, BPE tokenizer, sampler, TinyLlama end-to-end | 📅 Scheduled |
| **Module 8: Evaluation & Polish** | May – Jun 2027 | Python bindings, `llama-bench` comparison, final thesis | 📅 Scheduled |

## The development setup

| | Machine A (laptop) | Machine B (Development GPU Box) |
|---|---|---|
| GPU | None / Integrated | RTX 4070 SUPER (sm_89, 56 SMs, 504 GB/s) / RTX 4090 |
| Environment | Host OS / Linux | WSL2 Ubuntu 24.04 (nvcc 12.6, Driver 572.16 / 13.2) |
| Purpose | Write C++, CPU tests, docs | Compile CUDA, GPU tests, benchmarks, profiling |
| CUDA? | `ENGINE_CUDA_ENABLED=OFF` | `ENGINE_CUDA_ENABLED=ON` |
| Transport | Git push | Git pull |

**RULE:** Never edit code directly on Machine B without committing. Write, push, pull on B, test. A benchmark
whose source isn't the committed source is unreproducible. Git commit hash is automatically baked into the benchmark executable.

## Tech stack

- **C++17** — the engine, tensor library, CPU reference implementations
- **CUDA** — all GPU kernels (`.cu` files under `kernels/`)
- **CMake** — build system, CUDA-optional detection
- **Python 3** — `tools/gen_reference.py` (float64 golden data via PyTorch/NumPy)
- **No external ML libraries in the inference path** — everything hand-written

## Hard constraints (do not violate)

1. **No PyTorch/cuBLAS/ONNX in the engine.** They exist only in `tools/` and `bench/`.
2. **CUDA is optional.** The project must configure, build, and pass its CPU test suite
   without an NVIDIA toolchain. See ADR 0001.
3. **Correctness before performance.** `ctest` before `bench_kernels`, every time.
4. **Three-tier verification.** Tier 1 (float64 reference) → Tier 2 (CPU oracle) →
   Tier 3 (CUDA kernel). Tier 3 is checked against Tier 1, not Tier 2.
5. **Tolerances are derived, not tuned.** Every tolerance carries its arithmetic in a
   comment. "Widened until green" is how a real bug ships.
6. **Predict before you measure.** Write the prediction in the lab notebook *before*
   running the benchmark.
7. **One exercise per commit.** With the test result and benchmark delta in the message.
8. **Never edit a past lab notebook entry's numbers.** Add a note to the current week.

## Coding style

- **C++17** standard, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`
- **Naming:** `snake_case` for functions/variables, `PascalCase` for types,
  `kCamelCase` for constants, `SCREAMING_CASE` for macros
- **Comments:** every tolerance gets its arithmetic; every design choice gets its
  reasoning. When in doubt, write the comment. A month from now you will not remember.
- **Clang-format:** see `.clang-format` at the project root.
- **Error handling:** `ENGINE_CHECK()` macro, throws `EngineError`. CUDA errors use
  `CUDA_CHECK()` and `CUDA_CHECK_KERNEL()`.

## Key build commands

```bash
# Machine A — CPU-only
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
python3 tools/gen_reference.py
ctest --test-dir build --output-on-failure

# Machine B — with CUDA
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
build/bin/bench_kernels
```
