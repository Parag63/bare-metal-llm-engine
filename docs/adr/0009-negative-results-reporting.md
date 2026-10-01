# ADR 0009 — Negative Results Documentation and Architectural Boundary Tracking

**Status:** accepted · **Date:** 2026-10-01 · **Applies to:** `docs/negative-results.md`, `docs/lab-notebook.md`

## Context

In performance engineering, the set of optimizations that *do not work* often provides deeper architectural insight than those that do. Standard engineering practices frequently discard negative trials, leading to:
1. Future contributors repeating debunked optimizations.
2. Incomplete understanding of microarchitectural boundaries (such as L1/L2 cache sizing, bank conflict rules on Ada Lovelace, or warp scheduler behavior).
3. The mistaken assumption that textbook optimizations always yield positive results on modern hardware.

## Decision

We establish an explicit architectural requirement to document all non-viable, neutral, or negative optimization attempts in a dedicated living document: `docs/negative-results.md`.

For every negative result, the documentation must record:
1. **The Hypothesis:** What textbook principle or theoretical model predicted an improvement.
2. **The Implementation:** The concrete CUDA code or scheduling change applied.
3. **The Empirical Measurement:** The quantitative baseline vs modified performance (bandwidth, latency, cache hit rate).
4. **The Microarchitectural Root Cause:** Why the hardware behaved differently than the naive mental model (e.g. L2 cache hit masking, stride alignment, register pressure).
5. **The Architectural Boundary:** The exact operational envelope where the technique remains invalid or when it might become relevant again.

## Initial Documented Negative Results

1. **Shared Memory Bank Conflict Padding in Tiled GEMM:**
   - *Hypothesis:* Padding `__shared__ float s_A[32][33]` eliminates 2-way bank conflicts when reading columns.
   - *Result:* Slowdown from 2,563 GFLOP/s to 2,420 GFLOP/s (-5.6%).
   - *Root Cause:* Increased shared memory footprint degraded SM warp occupancy from 4 blocks to 3 blocks per SM, outweighing the minor latency savings of conflict-free reads.

2. **DRAM Bandwidth Savings from RMSNorm-Linear Fusion at $M=1$ Decode:**
   - *Hypothesis:* Fusing RMSNorm with Linear projection saves DRAM read/write traffic.
   - *Result:* Measured speedup is fixed at ~3–5 µs across varying $K$.
   - *Root Cause:* At $M=1$, intermediate activation tensor is only 16 KiB, residing entirely in the 48 MiB L2 cache. The speedup stems solely from launch overhead elimination, not memory bandwidth reduction.

3. **Over-Partitioning Grid for GEMV:**
   - *Hypothesis:* Increasing blocks by using 16 columns/block instead of 64 columns/block improves load balancing.
   - *Result:* Bandwidth dropped from 473.5 GB/s to 382.1 GB/s (-19.3%).
   - *Root Cause:* Increased atomic/memory tail effect and reduced instruction-level parallelism per warp.

## Consequences

- The project maintains an honest, scientifically grounded engineering record.
- Prevents architectural regressions and wasted engineering cycles.
- Enhances academic project defense by demonstrating rigorous microarchitectural analysis rather than trial-and-error tuning.
