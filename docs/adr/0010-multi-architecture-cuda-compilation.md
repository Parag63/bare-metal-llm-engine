# ADR 0010 — Flexible CUDA Architecture Compilation for Native Hardware Execution

**Status:** accepted · **Date:** 2026-10-01 · **Applies to:** `CMakeLists.txt`, `cmake/EngineCuda.cmake`, `docs/lab-notebook.md`

## Context

The engine targets NVIDIA Ada Lovelace architecture (`sm_89`), with empirical verification conducted on the NVIDIA GeForce RTX 4070 SUPER (56 SMs, 12 GB GDDR6X, 504 GB/s peak BW) alongside the RTX 4090 (128 SMs, 24 GB GDDR6X, 1008 GB/s peak BW) reference.

In [ADR 0005](0005-cuda-arch-explicit.md), `ENGINE_CUDA_ARCH` was established as an explicit CMake setting defaulting to `"89"` to prevent silent driver PTX JIT compilation and guarantee that performance numbers reflect native machine code (SASS). However, hardcoding a single fixed scalar value in the build configuration prevented flexibility for cross-compiling or targeting multiple architectures when distributing binaries.

`nvcc` supports compiling binaries containing native SASS machine code for specified compute targets:
```bash
-gencode arch=compute_89,code=sm_89
```

## Decision

1. **Permit Flexible Architecture Specification:**
   - `ENGINE_CUDA_ARCH` supports semicolon-separated architecture lists in CMake (e.g., `"89"` or multi-arch lists).
   - The default remains `"89"` for focused local builds targeting Ada Lovelace hardware.
2. **Per-Architecture Verification:**
   - `cmake/EngineCuda.cmake` validates all specified architectures against `nvcc --list-gpu-arch`. If an architecture is unsupported by the host CUDA toolkit, a clear warning is emitted with remediation steps.
3. **Dynamic Hardware Property Queries:**
   - All runtime microarchitectural metrics (peak bandwidth, SM counts, warp sizes, live SM clocks via NVML, L2 cache capacity) are queried dynamically at runtime via `cudaGetDeviceProperties` and NVML, rather than being hardcoded to a specific GPU model.
4. **Conditional Architecture Optimizations:**
   - Kernel code uses compile-time `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)` when leveraging Ada-specific hardware features, while preserving portable fallbacks.

## Consequences

- **Zero PTX JIT Overhead:** The resulting binary contains native binary instructions for the target architecture (`sm_89`). The CUDA driver executes native SASS directly, eliminating runtime JIT latency and avoiding non-native execution penalties.
- **Reproducible Benchmarking:** The benchmark harness provenance records both the compiled architectures (`ENGINE_CUDA_ARCH_STRING`) and the runtime GPU model, ensuring complete traceability in benchmark outputs and the lab notebook.
- **Clean Developer Workflow:** Machine A (CPU laptop) remains free of GPU build requirements, while Machine B builds native code tailored to the GPU.
