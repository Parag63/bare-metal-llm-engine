# Lab notebook

One entry per week, newest at the top. This file is the primary evidence of how the
project progressed, and it exists for three specific readers:

1. **Your mentor, at the weekly checkpoint.** The template below is ordered so that the
   answers to "what did you do, does it work, how do you know" are the first three
   headings.
2. **You, in June 2027,** writing the final report. Benchmark numbers you did not record
   at the time cannot be reconstructed, because the code will have changed.
3. **An interviewer.** "I predicted 16x from a 16-wide tile and measured 11x, and the
   gap was launch overhead plus partial edge tiles" is a much stronger answer than a
   table of results, and it only exists if you wrote the prediction down *before*
   measuring.

## Rules that make this worth keeping

- **Write the prediction before the measurement.** A benchmark without a prior
  expectation teaches you almost nothing; the same benchmark against a written
  prediction either confirms a mental model or shows exactly where it is wrong.
- **Paste the whole table, including the footer.** `bench_kernels` emits its device
  name, theoretical peak bandwidth, CUDA arch, build type and timestamp for this reason.
  A table without provenance is not comparable to any other table.
- **Record the locked GPU clock.** Two runs at different clocks are not comparable and
  you will not remember which was which. If you did not lock the clocks, write that
  down too — it is the honest thing and it explains the spread.
- **Record what did *not* work.** A kernel variant that turned out slower is a result.
  It is also the answer to "did you consider X?", which you will be asked.
- **Never edit a past entry's numbers.** If a measurement turns out to have been wrong,
  add a note to the current week explaining why. The history of what you believed and
  when is part of the record.

## Copy this template

```markdown
## Week NN — YYYY-MM-DD

**Objective / module:** (e.g. Objective 2, exercise 6 — tiled matmul)

### What I did

### Does it work

  ctest --test-dir build --output-on-failure -R kernels
  passed __   failed __   pending __   skipped __

Newly promoted from TEST_PENDING to TEST:
  -

### Prediction, written before measuring

### Measurement

GPU clock: locked to ____ MHz  (nvidia-smi -pm 1 && nvidia-smi -lgc ____)
Build type: RelWithDebInfo / Release

<paste the full bench_kernels table here, footer included>

### Prediction vs measurement — what the gap was

### What did not work

### Open questions for the mentor

### Next week
```

---

## Week 00 — 2026-08-26 · Scaffolding

**Objective / module:** project setup, before Objective 1 begins.

### What I did

Built the repository skeleton and the two verification harnesses that everything else
will be judged against.

- CUDA-optional CMake build ([ADR 0001](adr/0001-cuda-optional-build.md)). The whole
  project configures, builds and passes its CPU suite with no NVIDIA toolchain, so the
  laptop remains a real development machine.
- Three-tier correctness harness: `tools/gen_reference.py` writes float64 golden files;
  `src/cpu_ref/` is a scalar C++ oracle checked against them; CUDA kernels are checked
  against the *golden files*, not against the CPU oracle, so a misunderstanding shared by
  both implementations cannot pass.
- Custom test harness (`tests/test_framework.hpp`) with `TEST_PENDING` for unwritten
  modules, runtime `SKIP_TEST` for missing prerequisites, and stub-sentinel detection so
  a not-implemented kernel cannot pass a test vacuously.
- Benchmark harness (`bench/bench_harness.hpp`): CUDA-event timing, warmup, median/min
  with a spread column, ideal-traffic byte accounting, arithmetic intensity, and a
  provenance footer on every table.
- Exercise 1 (`vector_add`) written as a fully-worked reference; exercises 2–6 stubbed
  with specs, algorithm sketches, pitfalls and acceptance criteria.

### Does it work

On the laptop (CPU-only build):

```
passed 30   failed 0   pending 34   skipped 0
```

30 real passes across `dtype` (10), `golden` (6), `cpu_ref` (10) and `storage` (4). The
34 pending are Module 1's tensor specification plus the ten unwritten-kernel tests — the
correct state, not a failure. Warning-free under `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion`.

Not yet run on the 4090 box. Six `kernels` tests should pass there immediately —
`vector_add` correctness plus the launcher-contract tests — and ten should report
pending.

### Prediction, written before measuring

Nothing to predict yet; no kernel has been optimised. The numbers below are the
denominator, not a result.

### Measurement

GPU clock: n/a — CPU-only machine.
Build type: `-O2 -DNDEBUG`

```
### CPU reference baselines (single-threaded scalar, FP64 accumulators)

| kernel | size | median (ms) | min (ms) | spread | GFLOP/s | GB/s | % peak BW | AI (FLOP/B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cpu::vector_add` | 12.0 MiB | 1.88 | 1.81 | 37.02% (!) | 0.559 | 6.71 | -- | 0.083 |
| `cpu::vector_add` | 192.0 MiB | 27.51 | 23.94 | 47.70% (!) | 0.610 | 7.32 | -- | 0.083 |
| `cpu::reduce_sum` | 64.0 MiB | 26.23 | 25.22 | 11.02% (!) | 0.640 | 2.56 | -- | 0.250 |
| `cpu::softmax_rows` | 8 x 50257  (GPT-2 logits) | 6.18 | 5.71 | 32.23% (!) | 0.325 | 0.520 | -- | 0.625 |
| `cpu::rmsnorm` | 512 x 4096  (Llama-2-7B hidden) | 5.48 | 5.37 | 9.18% | 1.53 | 3.06 | -- | 0.500 |
| `cpu::matmul` | 256^3 | 30.56 | 29.81 | 10.71% (!) | 1.10 | 0.026 | -- | 42.67 |
| `cpu::matmul` | 512^3 | 574.5 | 567.7 | 7.62% | 0.467 | 0.0055 | -- | 85.33 |

Hardware: no CUDA device (CPU-only build)
Build: optimised (NDEBUG set), CPU-only build, Aug 26 2026 20:00:14
```

These are provisional: they were taken on a shared, virtualised machine, and five of
seven rows are flagged `(!)` for spread above 10%. Re-run on real hardware before quoting
any of them. That the spread column caught it on the first live run is the point of
having the column.

### Prediction vs measurement — what the gap was

One thing worth noting, because it was not planned: `cpu::matmul` reaches 1.10 GFLOP/s at
256³ and only 0.467 at 512³. Same code, half the throughput. That is the working set
falling out of cache — three 256² float matrices are 768 KiB and fit in L2; three 512²
are 3 MiB and do not. The naive triple loop's column-major access to `B` is what makes it
so sensitive. This is exactly the effect tiling fixes in exercise 6, showing up on the CPU
first.

### What did not work

- The first `tests/CMakeLists.txt` used
  `FAIL_REGULAR_EXPRESSION "passed 0 "` to catch a ctest entry that runs no tests. It
  would have failed a *correct* run: `--filter=tensor.` legitimately reports
  `passed 0 failed 0 pending 34`. Replaced with an exit code 2 from the harness when a
  non-empty filter selects zero tests, which distinguishes "no tests ran" from "no tests
  passed".
- Defining `ENGINE_ALWAYS_SYNC_CHECK` on the test *executable* does nothing.
  `CUDA_CHECK_KERNEL()` is expanded inside `kernels/*.cu`, which compile into the
  `engine` library, so the macro was already expanded by the time the test binary was
  compiled. Replaced with a library-wide `ENGINE_SYNC_CHECK_KERNELS` option, default off
  because per-launch synchronisation invalidates every benchmark.
- Forwarding `-Xcompiler=-Wall` through `nvcc` unconditionally is wrong on Windows: MSVC
  reads `-Wall` as `/Wall` and emits thousands of warnings from the Windows SDK. Since the
  4090 box is a Windows machine, that was the branch that would have run. Now branches to
  `/W4` under MSVC.
- A fixed one-decimal format in the benchmark table printed a compute-bound matmul's
  ideal bandwidth as `0.0 GB/s`, which looks like a harness bug rather than the correct
  answer. Now uses magnitude-dependent precision.

### Open questions for the mentor

- Target model: TinyLlama 1.1B for development and quantized Llama-2-7B as the headline
  result? The loader is being kept architecture-agnostic either way.
- Is the three-tier verification (float64 reference → CPU oracle → CUDA kernel, with tier
  3 compared to tier 1) the right level of rigour, or excessive for the timeline?

### Next week

Run the suite and `bench_kernels` on the 4090 box to establish the real baseline table —
including the launch-overhead figure, which every later fusion decision depends on. Then
exercise 2, `reduce_sum`: the deterministic two-stage version first, then the `atomicAdd`
variant, comparing both speed and run-to-run reproducibility.
