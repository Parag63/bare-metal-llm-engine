# ADR 0004 — cuBLAS is a benchmark baseline, never an implementation

**Status:** accepted · **Date:** 2026-08-26 · **Enforced by:** `bench/CMakeLists.txt`,
`cmake/EngineCuda.cmake`

## Context

cuBLAS ships with the CUDA toolkit, is already available on the 4090 box, and its
`cublasSgemm` is faster than any GEMM this project will hand-write. Linking it would
immediately give the engine a production-quality matmul.

Objective 2 of this project is to hand-write GEMM.

## Decision

cuBLAS is linked into **exactly one target**: `bench_kernels`. It is used for exactly one
purpose: to answer "how close did my kernel get?".

- `cmake/EngineCuda.cmake` calls `find_package(CUDAToolkit)` for `CUDA::cudart` and
  notes explicitly that cuBLAS is not linked into the engine.
- `bench/CMakeLists.txt` adds `CUDA::cublas` to `bench_kernels` only, behind the
  `ENGINE_BENCH_CUBLAS` option.
- The engine library links `CUDA::cudart` and nothing else.

If `CUDA::cublas` ever appears in a target other than `bench_kernels`, the point of the
project has been lost.

## Consequences

**The comparison is the deliverable.** A hand-written kernel measured against nothing is
an anecdote. `bench_kernels` prints `matmul_naive`, `matmul_tiled` and
`cublasSgemm (baseline)` for the same shapes on the same run, so the tiled kernel's
result is stated as a fraction of what the hardware can actually do. Reaching 40–60% of
cuBLAS with a hand-written FP32 kernel is a good result; that framing is only available
if cuBLAS is measured.

**It is not a fair fight, and saying so is part of the work.** cuBLAS uses tensor cores,
hand-tuned SASS, and shape-specific kernel selection. It is not the target to beat and
pretending otherwise would be dishonest in the report. It is the yardstick.

**A missing cuBLAS degrades, it does not break.** cuBLAS is a separate component of the
toolkit and a minimal install can have `cudart` without it. `bench/CMakeLists.txt` guards
on `if(TARGET CUDA::cublas)` and falls back to a warning, consistent with the rest of the
build system where an absent optional dependency never stops you working.
`ENGINE_BENCH_CUBLAS` is always defined to `0` or `1` — never left undefined — so
`#if ENGINE_BENCH_CUBLAS` cannot silently evaluate an undefined identifier and make the
baseline rows vanish without explanation.

**One implementation detail that costs an hour if you get it wrong.** cuBLAS is
column-major and this project is row-major. A row-major M×K matrix occupies the same
bytes as a column-major K×M matrix, so reading our `A` as a cuBLAS matrix yields `Aᵀ`.
Using `Cᵀ = Bᵀ · Aᵀ`, and noting that a column-major read of our row-major `C` *is* `Cᵀ`,
the correct call swaps the operands and passes `CUBLAS_OP_N` for both:

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
            B, N,   // our B, read column-major
            A, K,   // our A, read column-major
            &beta, C, N);
```

Passing `CUBLAS_OP_T` instead also produces the right answer but takes a slower path
inside cuBLAS — which would understate the baseline and flatter our kernel. A fair
baseline requires the operand swap, not just a tidy one.

## Why not the alternatives

**Link cuBLAS and use it for GEMM.** Deletes the project's second objective. Also removes
the reason to learn tiling, occupancy, and shared-memory bank conflicts, which is the
transferable skill the whole exercise ladder exists to build.

**Use cuBLAS as a temporary placeholder until the hand-written kernel is ready.** The
predictable outcome is that the placeholder stays: the engine works, the tests pass, and
the hand-written kernel becomes optional. Keeping cuBLAS out of the library means the
engine does not run until the real kernel exists, which is the correct forcing function.

**Compare against a published FLOP/s figure instead of measuring cuBLAS locally.**
Published numbers come from different clocks, different toolkit versions, and often
tensor-core paths. Measuring both on the same machine in the same run, at a locked clock,
is the only comparison that supports a claim.
