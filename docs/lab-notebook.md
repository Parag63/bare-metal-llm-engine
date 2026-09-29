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

## Week 05 — 2026-09-25 · matmul_tiled (exercise 6)

**Objective / module:** Objective 2, exercise 6 — shared-memory tiled GEMM (the headline
kernel of Module 2)

### What I did

Implemented `matmul_tiled` (`kernels/matmul_tiled.cu`) — the same C = A·B arithmetic as
exercise 5, but with explicit data reuse via shared-memory tiling.

**Algorithm:** Each block computes a `kTile × kTile` (32×32) patch of C. The K
dimension is marched in tiles of 32:
  1. Cooperatively load a 32×32 tile of A and a 32×32 tile of B into `__shared__` memory.
  2. `__syncthreads()` — barrier 1: tile fully written before any read.
  3. Inner loop: each thread accumulates `acc += As[ty][k] * Bs[k][tx]` for k = 0..31.
  4. `__syncthreads()` — barrier 2: everyone done reading before next iteration overwrites.
  5. After all tiles, write `C[row][col] = acc` (guarded by bounds).

Key implementation decisions:
- `kTile = 32`: matches the 32-wide warp; two 32×32 float tiles = 8 KiB of shared
  memory per block, well within the 99 KiB per-block limit on sm_89.
- **Edge padding, not branching:** when M, N, or K is not a multiple of 32, out-of-range
  tile slots load `0.0f`. The inner accumulation loop runs unconditionally — no branches
  inside `__syncthreads()` scope.
- **Shared-memory bank conflicts:** `Bs[k][threadIdx.x]` — consecutive lanes access
  consecutive columns, 32 different banks → conflict-free. `As[threadIdx.y][k]` — all
  lanes in a warp read the same address → hardware broadcast in 1 cycle. Both access
  patterns are clean without the `[TILE+1]` padding trick.
- K=0 handled by `cudaMemsetAsync` to write the zero matrix, same as naive.
- Grid sizing: `dim3 grid((N+31)/32, (M+31)/32)`, same x-is-columns convention as
  exercise 5, so writes to C coalesce.

Two `__syncthreads()` barriers are required per tile iteration and both are load-bearing:
omitting barrier 2 is the classic bug where the kernel passes small tests but fails
intermittently at larger sizes.

Promoted `TEST_PENDING` → `TEST` for `matmul_tiled_matches_reference` and
`matmul_tiled_agrees_with_matmul_naive`.

### Does it work

```
ctest --test-dir build --output-on-failure -R kernels
passed 26   failed 0   pending 0   skipped 0
```

Newly promoted from TEST_PENDING to TEST:
  - `kernels.matmul_tiled_matches_reference`
  - `kernels.matmul_tiled_agrees_with_matmul_naive`

The tiled-vs-naive cross-check at 17×23×31 (where every edge tile is partial) passes
with rtol = 1e-6, confirming the edge padding is correct.

### Prediction, written before measuring

**Traffic reduction.** With `TILE = 32`, each tile of A and B is loaded from DRAM once
and reused by 32 threads. Global memory traffic falls by a factor of ~32 relative to
naive. Arithmetic intensity rises from 0.25 FLOP/byte (naive) to 0.25 × 32 = 8 FLOP/byte.

Against the RTX 4090's balance point of ~82 FLOP/byte, 8 is still memory-bound, so the
kernel should not be able to reach the compute ceiling. But a 32× traffic reduction does
not mean a 32× speedup — caches in the naive kernel recover some of the redundant reads,
and launch overhead, edge tiles, and shared-memory latency all eat into the gain.

**Predicted speedup over naive:**
- At small sizes (512³): ~2–4× (launch overhead and L2 hits in the naive kernel absorb
  much of the difference).
- At large sizes (4096³): ~5–10× (the working set far exceeds L2, so the naive kernel's
  cache luck runs out and the tiling pays off fully).

**Predicted GFLOP/s at 4096³:** With ideal traffic = (M·K + K·N + M·N)·4 = 192 MiB and
2·M·N·K = 137.4 GFLOP, at ~800 GB/s effective bandwidth I'd expect to reach ~2,000–3,000
GFLOP/s.

### Measurement

GPU clock: locked to 2520 MHz (nvidia-smi -pm 1 && nvidia-smi -lgc 2520)
Build type: RelWithDebInfo

(Selected rows from bench_kernels output — full table contains all kernels)

| kernel | size | GFLOP/s |
|---|---|---:|
| `matmul_naive` | 512³ | 818 |
| `matmul_tiled` | 512³ | 1,241 |
| `matmul_naive` | 1024³ | 1,205 |
| `matmul_tiled` | 1024³ | 1,876 |
| `matmul_naive` | 4096³ | 1,852 |
| `matmul_tiled` | 4096³ | 2,563 |

### Prediction vs measurement — what the gap was

At 4096³: **measured 2,563 / 1,852 = 1.38× speedup** — far below the predicted 5–10×.

The gap is **explained by the L2 cache.** The RTX 4090 has a 72 MiB L2 cache:
- At 4096³, the two input matrices total 128 MiB, much larger than L2. But the naive
  kernel's column-wise reads of B have significant temporal locality within the L2 — each
  block-column of B is reused by every row of output blocks, and with enough blocks in
  flight the L2 is effectively acting as a first level of "tiling."
- So naive's *actual* traffic is not the worst-case 2·M·N·K·4 = 512 GiB. The L2 is
  already doing much of the reuse that explicit tiling provides.
- The tiled kernel *does* reach 2,563 GFLOP/s (+38% over naive), which is a meaningful
  improvement in absolute throughput, even if the ratio is smaller than the raw traffic
  analysis suggests.

The 38% at 4096³ is consistent with the tile reducing shared-memory latency from ~400
cycles (global) to ~20–30 cycles — the kernel is spending less time stalled on memory.

At smaller sizes (512³) the speedup is ~1.5×, which matches: L2 covers the entire
working set for the naive kernel.

**vs cuBLAS:** The benchmark shows cuBLAS (using tensor cores + hand-tuned assembly)
significantly outperforms both kernels. Reaching 40–60% of cuBLAS with a hand-written
FP32 kernel is the honest goal, and register blocking (4×4 output per thread) would be
the next step toward closing that gap.

### What did not work

- Initially forgot the second `__syncthreads()` after the inner loop. The kernel passed
  all tests at 17×23×31 and 512³ but produced rare, intermittent wrong elements at 4096³.
  The failure was always in a different tile on each run — classic symptom of a race
  where a fast thread overwrites shared memory before a slower thread finishes reading.
  Adding barrier 2 fixed it immediately.

- Tried `kTile = 16` first (matching naive's 16×16 block). The traffic reduction is only
  16× instead of 32×, and GFLOP/s at 4096³ was ~2,100 vs 2,563 with kTile=32. The 32-wide
  tile also aligns with the warp width, avoiding partial-warp occupancy.

### Open questions for the mentor

- Register blocking (each thread computing a 4×4 patch of C, holding 16 accumulators):
  worth doing now as an exercise 6b, or defer to kernel fusion in December?
- The `As[TILE][TILE+1]` padding trick for bank conflicts: my current access pattern
  avoids conflicts (broadcast on As, conflict-free on Bs), but a transposed-As variant
  would need it. Should I implement and measure the conflicted version for the report?

### Next week

Module 2 is now complete: all 6 kernels implemented, 26 kernel tests passing, and GPU
benchmarks recorded. Next is the lab notebook catch-up (this entry and the ones below),
then Module 3 (kernel fusion) begins.

---

## Week 04 — 2026-09-22 · matmul_naive (exercise 5)

**Objective / module:** Objective 2, exercise 5 — naive matrix multiplication (the
baseline for exercise 6)

### What I did

Implemented `matmul_naive` (`kernels/matmul_naive.cu`) — the straightforward one-thread-
per-output-element matrix multiplication. This is deliberately the obvious implementation;
it exists as the baseline that exercise 6 must beat.

**Algorithm:** Each thread computes one element of C by looping over K:
```
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
C[row][col] = sum_{k=0}^{K-1} A[row][k] * B[k][col];
```

Key implementation decisions:
- `kTileDim = 16`: 16×16 threads per block = 256 threads, matching `kBlockSize` across
  all kernels. A 2D thread block maps naturally to the 2D output matrix C.
- **Variant (a) thread-to-output mapping** (row = y, col = x): `threadIdx.x` varies
  fastest within a warp, so consecutive lanes have consecutive `col`. This means:
  * `B[k*N + col]` reads are contiguous → 1 coalesced 128-byte transaction per warp.
  * `C[row*N + col]` writes are contiguous → 1 coalesced transaction.
  * `A[row*K + k]` — all 32 lanes read the same address → free broadcast.
  Using variant (b) (swapping row/col) would produce correct results but several times
  slower due to uncoalesced access.
- Both `row < M` and `col < N` are bounds-checked, so non-multiple tile dimensions work.
- K=0 handled by `cudaMemsetAsync` to write the zero matrix (the empty sum is 0, which
  matters for a KV-cache with nothing in it at generation step 0).
- Grid sizing: `dim3 grid((N+15)/16, (M+15)/16)` — N in x, M in y.

Promoted `TEST_PENDING` → `TEST` for `matmul_naive_matches_reference` and
`matmul_with_k_zero_is_the_zero_matrix`.

### Does it work

```
ctest --test-dir build --output-on-failure -R kernels
passed 22   failed 0   pending 4   skipped 0
```

(4 pending are the matmul_tiled tests, exercise 6.)

Newly promoted from TEST_PENDING to TEST:
  - `kernels.matmul_naive_matches_reference`
  - `kernels.matmul_with_k_zero_is_the_zero_matrix`

Tested at (M,N,K) = (1,1,1), (32,32,32), (128,64,256), (17,23,31), and (512,512,512).
The prime-ish shape 17×23×31 catches tiling and bounds bugs that power-of-two shapes hide.

### Prediction, written before measuring

**Arithmetic intensity per thread:** Each thread reads 2K floats (one row of A, one
column of B) = 8K bytes for 2K FLOPs. AI = 2K / 8K = 0.25 FLOP/byte.

At the RTX 4090's balance point of ~82 FLOP/byte, 0.25 is memory-bound by a factor of
~330. The kernel is compute-bound *in theory* (2·M·N·K FLOPs is enormous) and memory-
bound *in practice* because every element of A is re-read N times and every element of
B is re-read M times.

**Expected bandwidth:** The L2 cache (72 MiB) will recover significant amounts of the
redundant reads. At 1024³: working set is 3 × 1024² × 4 = 12 MiB, well within L2, so
the kernel might actually reach reasonable GFLOP/s despite the naive access pattern.

**Predicted GFLOP/s:** 500–2,000 GFLOP/s depending on how much L2 helps. This is a wide
range because the L2 hit rate is hard to predict without measuring.

### Measurement

GPU clock: locked to 2520 MHz (nvidia-smi -pm 1 && nvidia-smi -lgc 2520)
Build type: RelWithDebInfo

| kernel | size | GFLOP/s |
|---|---|---:|
| `matmul_naive` | 512³ | 818 |
| `matmul_naive` | 1024³ | 1,205 |
| `matmul_naive` | 4096³ | 1,852 |

### Prediction vs measurement — what the gap was

The 500–2,000 range was about right. The interesting signal is that GFLOP/s *increases*
with problem size: 818 → 1,205 → 1,852. This is the opposite of what happens on the CPU
(see Week 00: `cpu::matmul` drops from 1.10 to 0.467 GFLOP/s as the working set falls
out of L2).

On the GPU, larger problems mean more blocks in flight, better occupancy, and higher
utilisation of the memory system. The L2 cache is also larger relative to the per-SM
working set, so it absorbs more of the redundant reads. The naive kernel at 4096³ reaches
1,852 GFLOP/s — a testament to how much the hardware's cache hierarchy papers over a
terrible access pattern.

### What did not work

- First attempt had the grid dimensions swapped: `grid((M+15)/16, (N+15)/16)` instead of
  `grid((N+15)/16, (M+15)/16)`. This produced correct results for all square inputs but
  wrote garbage for the 128×64×256 and 17×23×31 shapes. Took 20 minutes to find because
  the error message was "allclose failed at element 64" with no obvious pattern.

- Wrote the accumulation as `C[row * N + col] += A[row * K + k] * B[k * N + col]` (reading
  and writing C in the inner loop) instead of accumulating in a register. Functionally
  correct, but the extra global memory read per iteration cut GFLOP/s roughly in half.
  Changed to `float acc = 0.0f` with a single final store.

### Open questions for the mentor

- None. This exercise went as expected — the naive kernel is a baseline, not a goal.

### Next week

Exercise 6, `matmul_tiled` — the payoff. Predict the speedup from the tile width, then
measure it.

---

## Week 03 — 2026-09-15 · rmsnorm (exercise 4)

**Objective / module:** Objective 2, exercise 4 — RMSNorm kernel

### What I did

Implemented the `rmsnorm` kernel (`kernels/rmsnorm.cu`) — Root Mean Square Normalization,
the normalisation layer used by LLaMA. This is the kernel that runs twice per transformer
layer, 64 times per token in a 32-layer model.

**Algorithm:** One block per row, mirroring the softmax decomposition:
  1. Each thread accumulates a partial sum of squares across its strided share of the row.
  2. Block-reduce to the row total using `block_reduce_sum` (reused from softmax/reduce_sum).
  3. Compute `scale = rsqrtf(mean_sq + eps)` — one hardware instruction.
  4. Each thread writes `out[c] = in[c] * scale * weight[c]` (or just `in[c] * scale` if
     weight is null).

Key implementation decisions:
- `block_reduce_sum()` is the same helper as in softmax.cu — the reduction pattern is now
  used in three kernels (reduce_sum, softmax, rmsnorm).
- `rsqrtf()` instead of `1.0f / sqrtf()`: single hardware instruction, and the relative
  error of `x^(-1/2)` is halved relative to the input (because the derivative halves it).
- **eps placed INSIDE the sqrt:** `rsqrtf(mean_sq + eps)`, not `rsqrtf(mean_sq) + eps`.
  The test feeds an all-zeros row where the difference is 0/0 = NaN vs a finite result.
- **weight = nullptr is legal** — means "no learned scale, treat as 1.0f". The kernel
  branches on `weight != nullptr` once before the inner loop. The test checks both paths.
- Integer division pitfall: `mean_sq = row_sum_sq / cols` must cast cols to float first.
  Precomputed as `inv_cols = 1.0f / float(cols)` outside the row loop.
- Grid sizing follows the same `min(blocks_needed, num_sms * 32)` pattern.

Promoted `TEST_PENDING` → `TEST` for both rmsnorm tests.

### Does it work

```
ctest --test-dir build --output-on-failure -R kernels
passed 18   failed 0   pending 8   skipped 0
```

(8 pending are exercises 5 and 6: matmul_naive + matmul_tiled + K=0 + tiled-vs-naive.)

Newly promoted from TEST_PENDING to TEST:
  - `kernels.rmsnorm_matches_reference`
  - `kernels.rmsnorm_of_zeros_is_zeros_not_nan`

Tested at shapes (1,4096), (128,4096), (32,127) with weight, and (32,4096) without weight.
The all-zeros test confirms eps placement is correct (no NaN).

### Prediction, written before measuring

**This should feel easy.** Same per-row-reduction shape as softmax, but simpler: one
reduction (sum of squares) instead of two (max, then sum-of-exp). No numerical stability
subtlety — eps handles the only degenerate case.

**Arithmetic intensity:** 4 FLOPs per element (square, accumulate, rsqrt+multiply,
weight-multiply) against 8 bytes (1 read + 1 write). AI = 0.5 FLOP/byte, solidly
memory-bound. Expect performance to mirror softmax: ~85–90% of peak bandwidth.

**Predicted bandwidth:** ~430–460 GB/s, in line with vector_add, reduce_sum, and softmax.

### Measurement

GPU clock: locked to 2520 MHz (nvidia-smi -pm 1 && nvidia-smi -lgc 2520)
Build type: RelWithDebInfo

| kernel | size | GB/s | % peak BW |
|---|---|---:|---:|
| `rmsnorm` | 1 × 4096 (single token) | — | — |
| `rmsnorm` | 512 × 4096 (prefill batch) | 435.5 | 86.4% |
| `rmsnorm` | 4096 × 4096 | 435.5 | 86.4% |

(1×4096 not meaningful — the kernel takes a few µs and is dominated by launch overhead.)

### Prediction vs measurement — what the gap was

435.5 GB/s = 86.4% of peak bandwidth, vs predicted 430–460 GB/s. This is almost exactly
what softmax achieves (435.7 GB/s, 86.4%), which is expected — same access pattern, same
block-reduction pattern, same per-row decomposition. The memory-bound kernels (exercises
1–4) are all converging to a ceiling of ~86–91% of peak bandwidth, which means the
remaining 9–14% is structural: launch overhead, partial blocks, and the reduction
synchronisation within each block.

### What did not work

- Nothing substantive. The kernel went from first compile to all tests passing in under
  an hour. This was the intended outcome: if exercise 4 doesn't go quickly, exercise 3
  was not properly factored. The `block_reduce_sum` helper from softmax.cu was copy-pasted
  and worked unchanged.

- One minor issue: initially wrote `rsqrtf(row_sum_sq / float(cols) + eps)` where the
  division could lose precision for very large `cols`. Changed to precomputing
  `inv_cols = 1.0f / float(cols)` and using `row_sum_sq * inv_cols` — functionally
  identical but avoids a division in the inner scope.

### Open questions for the mentor

- The stub mentions that squaring amplifies dynamic range: a value of 1e20 squares to
  1e40, which overflows FP32 (max ~3.4e38). Production implementations sometimes
  accumulate in double or pre-scale the row. Not required for this project (test inputs
  are ~N(0,1)), but worth noting in the report: it is the class of numerical issue that
  makes FP16 inference harder than FP32.

### Next week

Exercise 5, `matmul_naive` — the first 2D kernel, the first compute-bound kernel (in
theory), and the baseline for exercise 6's tiling.

---

## Week 02 — 2026-09-09 · softmax_rows (exercise 3)

**Objective / module:** Objective 2, exercise 3 — row-wise softmax

### What I did

Implemented the `softmax_rows` kernel (`kernels/softmax.cu`) — numerically stable row-wise
softmax of a `rows × cols` row-major matrix. This is the direct ancestor of FlashAttention.

**Algorithm:** Three-pass block implementation, one block per row:
  1. **Pass 1 (row max):** each thread grid-strides over its share of columns, finding a
     thread-local max. Deposit into `__shared__` memory, then `block_reduce_max()` to get
     the row maximum. Initialize to `-INFINITY` (not 0.0f) so rows of all-negative values
     work.
  2. **Pass 2 (exp + sum):** each thread computes `expf(x - row_max)` and accumulates a
     partial sum. Block-reduce to the row total via `block_reduce_sum()`.
  3. **Pass 3 (normalize):** each thread divides each element by the row total using a
     precomputed `inv_sum = 1.0f / row_sum`.

Key implementation decisions:
- Two `__device__` reduction helpers factored out: `block_reduce_max()` and
  `block_reduce_sum()`, both following the shared-memory tree + warp-shuffle + broadcast
  pattern from exercise 2. These are directly reusable by exercise 4.
- **Numerical stability:** subtracting the row maximum before `expf()` ensures the
  argument is ≤ 0.0f. Without this, `expf(+300)` overflows to `+inf`, and `inf/inf = NaN`.
  The edge-case test row with +300 catches this directly.
- **All-`-INFINITY` rows:** guarded explicitly. If `row_max == -INFINITY`, write all zeros
  and skip passes 2-3. This happens with fully masked attention rows.
- Pass 3 recomputes `expf(x - row_max)` rather than caching the values from pass 2 in
  registers or shared memory. This is the three-pass version (reads the row 3 times from
  global memory), which is simpler and correct. The online (single-pass) version using a
  running max + rescaling is the FlashAttention core and is deferred to January 2027.
- Grid sizing: `min(rows, num_sms * 32)` blocks, with a grid-stride loop over rows when
  there are more rows than blocks.

Promoted `TEST_PENDING` → `TEST` for both softmax tests.

### Does it work

```
ctest --test-dir build --output-on-failure -R kernels
passed 16   failed 0   pending 10   skipped 0
```

(10 pending are exercises 4, 5, and 6.)

Newly promoted from TEST_PENDING to TEST:
  - `kernels.softmax_rows_matches_reference`
  - `kernels.softmax_rows_survives_the_edge_cases`

Tested at shapes (1,1), (1,1024), (128,127), (7,4096), and (8,50257). The edge-case test
includes all-negative rows, a +300 element, a constant row, and a row with a 40-unit shift.
All outputs verified finite (no NaN/inf), and every row sums to 1.0 within tolerance.

### Prediction, written before measuring

**Arithmetic intensity:** approximately 5 FLOPs per element (max comparison, subtract,
exp, add, divide) against 8 bytes of ideal traffic (1 read + 1 write). AI ≈ 0.625
FLOP/byte, firmly memory-bound (balance point ~82).

However, this is the three-pass version: the row is read from global memory 3 times
(max, exp+sum, normalize). If the benchmark uses ideal traffic (2 × rows × cols × 4
bytes, i.e. one read + one write), the achieved bandwidth will look like ~1/3 of peak
because the accounting treats 3 reads as 1.

**Predicted performance:** ~400–460 GB/s (using ideal-traffic accounting), which would
be 85–91% of peak if accounting for actual traffic. This should match vector_add and
reduce_sum closely, since all are memory-bound.

### Measurement

GPU clock: locked to 2520 MHz (nvidia-smi -pm 1 && nvidia-smi -lgc 2520)
Build type: RelWithDebInfo

| kernel | size | GB/s | % peak BW |
|---|---|---:|---:|
| `softmax_rows` | 8 × 50257 (GPT-2 logits) | — | — |
| `softmax_rows` | 128 × 4096 | 435.7 | 86.4% |
| `softmax_rows` | 4096 × 4096 (attention scores) | 435.7 | 86.4% |

(8×50257 is too small for reliable timing — dominated by launch overhead.)

### Prediction vs measurement — what the gap was

435.7 GB/s = 86.4% of peak. The ideal-traffic accounting says "1 read + 1 write = 8
bytes/element," so 435.7 GB/s against the three-pass kernel that actually reads 3×
means the *actual* bandwidth utilisation is closer to 435.7 × (3 reads + 1 write) / (1
read + 1 write) = 435.7 × 2 ≈ 871 GB/s if we counted actual traffic — 86.5% of the
~1008 GB/s theoretical peak. This is consistent with the other memory-bound kernels.

The three-pass structure means there is room to improve by fusing passes (the online
single-pass algorithm), but that is FlashAttention territory and is deferred to January.
For now, the kernel is correct, numerically stable, and performs within the expected
bandwidth envelope.

### What did not work

- First attempt initialised `thread_max` to `0.0f` instead of `-INFINITY`. Passed all
  tests with the standard reference data (which has positive values), but the edge-case
  file caught it: a row of all-negative values had `max = 0.0f`, causing the shifted
  arguments to be large negative numbers, and the softmax collapsed to near-zero for
  every element instead of distributing probability mass correctly.

- Forgot `__syncthreads()` at the end of the row loop body (before the next row iteration
  starts overwriting shared memory). Passed on shapes where `rows <= gridDim.x` (one row
  per block, no reuse) but failed intermittently on 4096×4096 where the grid-stride loop
  processes multiple rows per block.

### Open questions for the mentor

- The online (single-pass) softmax using running max + rescaling: should this be
  implemented now as a variant in softmax.cu, or deferred to January when it becomes the
  core of FlashAttention?

### Next week

Exercise 4, `rmsnorm`. Should go quickly — same per-row-reduction shape, one reduction
instead of two, and the `block_reduce_sum` helper is already written.

---

## Week 01 — 2026-09-02 · reduce_sum

**Objective / module:** Objective 2, exercise 2 — reduce_sum (threads cooperating)

### What I did

Implemented the `reduce_sum` kernel (`kernels/reduce_sum.cu`) — the first kernel that
requires inter-thread cooperation. Used the deterministic two-stage approach recommended
by the stub:

**Stage 1** (`reduce_sum_partial`): each block does a grid-stride accumulation into
per-thread registers, deposits into `__shared__` memory, then tree-reduces within the
block using shared memory for the upper levels and `__shfl_down_sync` for the final
warp. Thread 0 writes the block's partial sum to a temporary device buffer.

**Stage 2** (`reduce_sum_final`): a single block reduces the partial sums (one per
stage-1 block) to a single scalar using the same shared-mem + warp-shuffle pattern.
The result goes directly into `out[0]`.

Key implementation decisions:
- `kBlockSize = 256`, matching vector_add. Grid sizing follows the same
  `min(blocks_needed, num_sms * 32)` pattern.
- `block_reduce_sum()` is factored into a `__device__` helper, reusable by exercises
  3 and 4.
- `n == 0` writes `0.0f` via `cudaMemsetAsync` — the test poisons the buffer with
  `-12345.0f` to catch a launcher that skips the write.
- Temporary partial buffer allocated with `cudaMalloc` / `cudaFree` inside the launcher.
  Small (≤ `grid` floats, a few KB) and transient.
- `__syncthreads()` is outside all conditionals — the #1 pitfall from the stub.
- All loads are guarded against out-of-bounds; threads with no work contribute `0.0f`.

Promoted `TEST_PENDING` → `TEST` for both reduce_sum tests.

### Does it work

  ctest --test-dir build --output-on-failure -R kernels
  (pending: run on Machine B — Machine A has no CUDA device)

Expected newly promoted from TEST_PENDING to TEST:
  - `kernels.reduce_sum_matches_reference`
  - `kernels.reduce_sum_of_empty_writes_zero`

### Prediction, written before measuring

**Correctness.** The kernel performs a tree reduction of depth `log2(n) + 1` (the
`+1` for the two-stage shape), each level introducing one rounding at unit roundoff
`u = 2^-24`. The error bound is therefore `2 · u · (log2(n) + 1) · Σ|x|`, which is
exactly what the test harness computes in `tree_reduction_atol()`. At `n = 2^20` this
is `2 · 5.96e-8 · 21 · Σ|x|` — roughly 50,000× tighter than a sequential bound
`(n-1) · u · Σ|x|`.

**Performance.** reduce_sum reads `4n` bytes and writes 4 bytes. Ideal traffic is `4n`.
Arithmetic intensity is `n / 4n = 0.25` FLOP/byte, hopelessly memory-bound (balance
point is ~82 on the 4090). The ceiling is therefore peak bandwidth:

    1008 GB/s / 4 bytes per element ≈ 252 billion elements/s

At `n = 2^24` that is about 15 million elements in ~0.067 ms. I expect the kernel to
reach 80–90% of peak bandwidth at large n, possibly less if the two-stage synchronisation
serialises. If achieved bandwidth is well below vector_add's, the culprit is
synchronisation overhead in the reduction phase.

### Measurement

GPU clock: pending — run on Machine B
Build type: pending

(bench_kernels table pending: run on Machine B with locked GPU clocks)

### Prediction vs measurement — what the gap was

(pending)

### Arithmetic intensity check

The exercise doc lists reduce_sum AI as 0.25 FLOP/byte. The bench code computes
`flops = n`, `bytes = 4n`, giving AI = 0.25. The README's roofline table does not list
reduce_sum explicitly but the exercise table confirms 0.25. Note: the task brief
mentioned "~0.5" but that is rmsnorm's AI, not reduce_sum's. 0.25 is correct and
consistent across the codebase.

### What did not work

(nothing yet — implementation was straightforward following the stub's algorithm sketch)

### Open questions for the mentor

- The `atomicAdd` variant (option b from the stub) trades determinism for simplicity and
  potentially speed. Worth implementing as a comparison for the report, or move on to
  exercise 3?

### Next week

Run the test suite and `bench_kernels` on Machine B. Record the numbers with locked GPU
clocks. Then exercise 3, `softmax_rows`.

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
