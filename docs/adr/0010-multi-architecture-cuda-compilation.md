# ADR 0010 — Multi-Architecture CUDA Compilation (`86;89`) for Cross-Hardware Portability

**Status:** accepted · **Date:** 2026-10-01 · **Applies to:** `CMakeLists.txt`, `cmake/EngineCuda.cmake`, `docs/lab-notebook.md`

## Context

The engine is developed and validated across multiple distinct hardware environments:
- **Machine A (Workstation / Remote Rig):** NVIDIA RTX A4000 (Ampere architecture, compute capability `sm_86`, 16 GB GDDR6 VRAM with ECC, 448 GB/s peak bandwidth, 48 SMs).
- **Machine B (Primary Workstation):** NVIDIA GeForce RTX 4070 SUPER (Ada Lovelace architecture, compute capability `sm_89`, 12 GB GDDR6X VRAM, 504 GB/s peak bandwidth, 56 SMs).
- **Target Flagship:** NVIDIA GeForce RTX 4090 (Ada Lovelace architecture, compute capability `sm_89`, 24 GB GDDR6X VRAM, 1008 GB/s peak bandwidth, 128 SMs).

In [ADR 0005](0005-cuda-arch-explicit.md), `ENGINE_CUDA_ARCH` was established as an explicit setting defaulting to `"89"` to prevent silent driver PTX JIT compilation and guarantee that performance numbers reflect native machine code (SASS). However, requiring single-architecture binaries creates workflow friction when switching between the A4000 (`sm_86`) and Ada (`sm_89`), and prevents building unified distribution packages.

`nvcc` supports compiling fat binaries containing native SASS machine code for multiple compute targets using multi-architecture flags:
```bash
-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
```

## Decision

1. **Permit Multi-Architecture Specification:**
   - `ENGINE_CUDA_ARCH` supports semicolon-separated architecture lists in CMake, for example `"86;89"`.
   - The default remains `"89"` for focused local builds, with `"86;89"` fully supported for dual-machine deployment.
2. **Per-Architecture Verification:**
   - `cmake/EngineCuda.cmake` loops over all architectures in `${ENGINE_CUDA_ARCH}` and validates each against `nvcc --list-gpu-arch`. If any architecture is unsupported by the host CUDA toolkit, a loud warning is emitted with remediation steps.
3. **Dynamic Hardware Property Queries:**
   - All runtime microarchitectural metrics (peak bandwidth, SM counts, warp sizes, live SM clocks via NVML, L2 cache capacity) are queried dynamically at runtime via `cudaGetDeviceProperties` and NVML, rather than being hardcoded to a specific GPU model.
4. **Conditional Architecture Optimizations:**
   - Kernel code uses compile-time `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)` only when leveraging Ada-specific hardware features (such as FP8 Tensor Cores), while preserving portable Ampere fallbacks (`__CUDA_ARCH__ >= 860`).

## Consequences

- **Zero PTX JIT Overhead on Both Machines:** The resulting binary contains native binary instructions for both `sm_86` and `sm_89`. The CUDA driver loads the exact SASS for whichever GPU is physically present, eliminating runtime JIT latency and avoiding non-native execution penalties.
- **Reproducible Cross-Architecture Benchmarking:** The benchmark harness provenance records both the compiled architectures (`ENGINE_CUDA_ARCH_STRING`) and the runtime GPU model, enabling clean cross-architecture comparisons (e.g. A4000 vs 4070 SUPER) in the lab notebook.
- **Build Footprint:** Dual-architecture fat binaries are approximately ~40% larger on disk, but compilation time is parallelized across build cores.
