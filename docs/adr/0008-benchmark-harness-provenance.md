# ADR 0008 — Benchmark Harness Structured JSON Logging and Git Provenance

**Status:** accepted · **Date:** 2026-10-01 · **Applies to:** `bench/bench_harness.hpp`, `bench/bench_kernels.cu`, `bench/bench_cpu_ref.cpp`, `tools/generate_results_table.py`

## Context

In performance engineering and academic research, benchmark results are easily rendered invalid or untrustworthy if the hardware configuration, clock speeds, driver versions, or exact source code commits are not recorded alongside the numbers.

Previously:
1. Benchmarks printed a formatted markdown table to `stdout`.
2. Updating the project `README.md` or lab notebook required manual copying, which was error-prone and risked drift between the committed code and reported figures.
3. The git commit hash and device driver version were not captured inside the benchmark binary, leading to ambiguity when reproducing figures across machines or after multiple commits.

## Decision

We introduce a structured JSON output mode and automated documentation synchronization pipeline:

1. **`--json` Flag in Benchmark Executables:**
   - Both `bench_kernels` and `bench_cpu_ref` accept a `--json` CLI flag.
   - When specified, results are emitted as a structured JSON document:
     ```json
     {
       "device": "NVIDIA GeForce RTX 4070 SUPER",
       "sm_count": 56,
       "theoretical_bandwidth_gbs": 504.0,
       "driver_version": "13.2",
       "gpu_clock_mhz": 2475,
       "git_commit": "2183c50",
       "build_type": "RelWithDebInfo",
       "results": [
         {
           "kernel": "gemv",
           "size": "1x12288x4096",
           "median_ms": 0.426,
           "min_ms": 0.424,
           "spread_pct": 2.1,
           "gflops": 965.8,
           "bandwidth_gbs": 473.5,
           "pct_peak_bw": 94.0,
           "arithmetic_intensity": 0.25
         }
       ]
     }
     ```

2. **Git Commit Hash Baking in CMake:**
   - `CMakeLists.txt` queries `git rev-parse --short HEAD` during configuration and generates `cmake/engine_config.hpp.in`.
   - The constant `ENGINE_GIT_HASH` is compiled into the binary.

3. **Device Driver and Clock Query Utilities:**
   - `cuda_device.cpp` queries `cudaDriverGetVersion` and `cudaDeviceGetAttribute` for `cudaDevAttrClockRate` and `cudaDevAttrMemoryClockRate`.

4. **Automated Documentation Synchronization:**
   - A dedicated Python tool, `tools/generate_results_table.py`, runs the benchmark or ingests an existing JSON file, formats the markdown table with full provenance headers, and automatically updates the results block in `README.md`.

## Consequences

- Benchmarks are 100% reproducible and self-documenting.
- Markdown documentation cannot drift from binary output.
- Downstream tools (such as `tools/roofline_plot.py`) can consume structured JSON data directly to generate visualization artifacts without fragile text scraping.
