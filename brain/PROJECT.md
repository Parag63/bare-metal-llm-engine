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
| Language | C++17 (engine), CUDA (kernels), Python (tools) |
| Build | CMake ≥ 3.20 |
| Target GPU | NVIDIA RTX 4090 (sm_89 / Ada Lovelace) |
| Dev model | TinyLlama 1.1B |
| Headline model | Quantized Llama-2-7B |

## The two-machine setup

| | Machine A (laptop) | Machine B (4090 box) |
|---|---|---|
| GPU | None | RTX 4090 (sm_89) |
| Purpose | Write C++, CPU tests, docs | Compile kernels, GPU tests, benchmarks |
| CUDA? | `ENGINE_CUDA_ENABLED=OFF` | `ENGINE_CUDA_ENABLED=ON` |
| Transport | Git push | Git pull |

**RULE:** Never edit code on Machine B. Write on A, push, pull on B, test. A benchmark
whose source isn't the committed source is unreproducible.

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
