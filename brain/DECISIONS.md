# Decision Log

> A consolidated record of *why* things are the way they are. When an AI agent or a
> future contributor asks "why did you do X instead of Y?", the answer is here.
>
> Formal ADRs live in `docs/adr/`. This file captures both those and the smaller,
> informal decisions that don't warrant a full ADR but are important enough to record.

## Formal Architectural Decisions (ADRs)

| ID | Title | Summary |
|---|---|---|
| [ADR-0001](../docs/adr/0001-cuda-optional-build.md) | CUDA is optional | CUDA is detected, never required — the whole project builds and tests without a GPU |
| [ADR-0002](../docs/adr/0002-storage-tensor-split.md) | Storage/Tensor split | `Storage` (owns bytes) and `Tensor` (a view) are separate types, so slices, mmap'd weights, and reshapes are free |
| [ADR-0003](../docs/adr/0003-raw-pointer-kernel-api.md) | Raw pointer kernel API | Kernel launchers take raw pointers and dimensions, not `Tensor` — decouples the kernel ladder from Module 1 |
| [ADR-0004](../docs/adr/0004-cublas-baseline-only.md) | cuBLAS is baseline only | cuBLAS is a benchmark baseline, linked into exactly one target, never an implementation |
| [ADR-0005](../docs/adr/0005-cuda-arch-explicit.md) | Explicit CUDA arch | `ENGINE_CUDA_ARCH` is explicit (`89`), not auto-detected — mismatches warn loudly |
| [ADR-0006](../docs/adr/0006-kernel-fusion-strategy.md) | Kernel fusion strategy | Target transformer sub-layer boundaries (`RMSNorm+Linear`, `Residual+RMSNorm`) to eliminate intermediate DRAM traffic |
| [ADR-0007](../docs/adr/0007-gemv-decode-specialization.md) | GEMV decode specialization | Dedicated matrix-vector kernel for $M=1$ autoregressive decode, reaching 94.0% peak BW and beating cuBLAS |
| [ADR-0008](../docs/adr/0008-benchmark-harness-provenance.md) | Benchmark harness JSON & provenance | Structured `--json` export, device driver, hardware clock, and git commit hash tracking |
| [ADR-0009](../docs/adr/0009-negative-results-reporting.md) | Negative results reporting | Explicitly document optimizations that failed or degraded performance to preserve empirical boundaries |
| [ADR-0010](../docs/adr/0010-multi-architecture-cuda-compilation.md) | Flexible CUDA architecture compilation | Support flexible architecture specification with native SASS generation without PTX JIT overhead |

## Informal Decisions

### D-001: Custom test framework instead of GTest
**Date:** Aug 2026
**Decision:** Write a custom test harness (`test_framework.hpp`) instead of using
Google Test.
**Why:** Need `TEST_PENDING` (tests that run but report pending instead of failed),
`SKIP_TEST` for missing prerequisites (no GPU, no reference data), and stub-sentinel
detection (a not-implemented kernel cannot satisfy a test). GTest doesn't support these
out of the box. The custom harness is ~300 lines and exactly fits the workflow.

### D-002: CPU reference uses double accumulators
**Date:** Aug 2026
**Decision:** All CPU oracle reductions accumulate in `double`, not `float`.
**Why:** A float accumulator over a large array loses low-order bits progressively
(~1e-3 relative error at 1M elements). If the oracle used float, a correct GPU tree
reduction (which is actually *more* accurate than a sequential float sum) would appear
to "fail" against it. Double sidesteps the argument entirely.

### D-003: Deterministic two-stage reduction over atomicAdd
**Date:** Sep 2026 (Exercise 2)
**Decision:** `reduce_sum` uses a deterministic two-stage approach (partial sums, then
a second kernel), not `atomicAdd`.
**Why:** `atomicAdd` is simpler and one launch, but floating-point addition is not
associative — the result varies run-to-run depending on block scheduling. This makes
debugging nearly impossible when you're trying to determine if a numerical difference
is a real bug or just ordering noise. Determinism is worth the extra launch.

### D-004: Grid sizing: `min(blocks_needed, num_sms * 32)`
**Date:** Aug 2026
**Decision:** All kernels size the grid to fill the machine rather than match the input
size: `min(blocks_needed, num_sms * 32)`.
**Why:** 32 blocks per SM is a generous oversubscription that gives the scheduler plenty
of warps to hide memory latency with. Combined with grid-stride loops, the launch
configuration becomes independent of the input size, which makes it reusable and
tunable.

### D-005: `block_reduce_sum` factored into a `__device__` helper
**Date:** Sep 2026 (Exercise 2)
**Decision:** The within-block reduction (shared-memory tree + warp shuffle) is a
reusable `__device__` function, not inline code.
**Why:** The same reduction pattern is needed by exercises 3 (softmax), 4 (rmsnorm),
and eventually FlashAttention. Writing it once and verifying it once is cheaper than
debugging three copies.

### D-006: Tensor constructor does NOT zero memory
**Date:** Aug 2026
**Decision:** `Tensor(shape, dtype, device)` leaves memory uninitialized. Use
`Tensor::zeros()` if you need zeroes.
**Why:** Adding a zeroing pass to every allocation adds a full memory-bandwidth pass
over every weight buffer you're about to overwrite anyway. On a 14 GB FP16 model at
~1 TB/s, that's ~14 ms of pure waste per allocation.

### D-007: eps is INSIDE the sqrt for RMSNorm
**Date:** Aug 2026
**Decision:** `scale = rsqrtf(mean_sq + eps)`, not `rsqrtf(mean_sq) + eps`.
**Why:** Both appear in the wild and differ only for near-zero rows. The CPU oracle uses
this convention, so the CUDA kernel must match it or tests disagree at eps-scale
tolerances. The test deliberately feeds an all-zeros row to catch this.

### D-008: `ENGINE_SYNC_CHECK_KERNELS` is a library-wide option
**Date:** Aug 2026
**Decision:** The synchronize-after-launch flag is set on the `engine` library target
(PUBLIC), not on individual test/bench executables.
**Why:** `CUDA_CHECK_KERNEL()` is expanded inside `kernels/*.cu`, which compile into
the `engine` library. Defining the flag on the test executable has zero effect — the
macro was already expanded when the library was compiled. This was a real bug in Week 00.

### D-009: 64-column tiling with float2 coalescing for GEMV
**Date:** Oct 2026 (Exercise 7)
**Decision:** `gemv` maps 64 columns of $B$ per block, using 256 threads (8 warps). Each warp reduces $K$ cooperatively using 64-bit `float2` loads and warp shuffles before writing to shared memory.
**Why:** During autoregressive generation ($M=1$), 2D tiled GEMM wastes thread resources and achieves low occupancy because $M < TILE$. The dedicated GEMV kernel achieves 473.5 GB/s (94.0% peak bandwidth), outperforming cuBLAS by 13.7%.

### D-010: Automated README benchmark synchronization via Python parser
**Date:** Oct 2026
**Decision:** `tools/generate_results_table.py` parses the `--json` output of `bench_kernels` and directly updates the markdown results table in `README.md`.
**Why:** Prevents manual copy-paste drift between lab notebook runs and public documentation. Enforces git provenance and machine configuration headers automatically.

### D-011: L2 cache residency accounting for $M=1$ fusion speedup
**Date:** Oct 2026
**Decision:** Do not attribute $M=1$ kernel fusion speedups to DRAM traffic reduction in the lab notebook.
**Why:** At $M=1, K=4096$, the intermediate activation tensor is only 16 KiB. Modern Ada Lovelace GPUs have 48–72 MiB L2 cache, meaning the intermediate data stays in L2 even in unfused execution. The measured 3–5 µs speedup per projection is entirely due to eliminating kernel launch overhead, not DRAM bandwidth savings.

### D-012: PoolAllocator power-of-two size class bucketing and 256-byte alignment
**Date:** Oct 2026 (Phase 4)
**Decision:** `PoolAllocator` organizes memory allocations into power-of-two size class freelists (256 B to 1 GiB) with contiguous slab carving for new sizes, enforcing 256-byte alignment.
**Why:** Eliminates runtime `cudaMalloc` / `cudaFree` driver call overhead ($\sim 10\text{--}50\ \mu\text{s}$ per call). 256-byte alignment ensures that all tensor buffers satisfy hardware coalescing constraints and 128-bit vector memory instruction requirements. Passed the 100,000 cycles acceptance test with `num_driver_allocs == 1`.

### D-013: 2D Register-tiled GEMM with transposed shared memory s_A
**Date:** Oct 2026 (Phase 4)
**Decision:** `matmul_register_tiled` maps a $128 \times 128$ block tile to 256 threads ($16 \times 16$), where each thread computes an $8 \times 8$ sub-matrix in 64 registers using outer products over $BK=8$. Shared memory for matrix A is stored transposed: `s_A[BK][BM]`.
**Why:** Standard shared-memory tiling (`matmul_tiled`) saturates shared-memory bank bandwidth ($\text{AI}_{\text{smem}} = 0.25\text{ FLOP/byte}$). Register tiling raises shared-memory arithmetic intensity by $8\times$ to $2.0\text{ FLOP/byte}$. Transposing `s_A` enables conflict-free stride-1 column reads by warps. Achieves 16.17–17.55 TFLOP/s (6.7x speedup over tiled and 73% of cuBLAS).

### D-014: Fused single-pass SwiGLU with 128-bit vector instructions
**Date:** Oct 2026 (Phase 4)
**Decision:** Fused SwiGLU activation computes $\text{SiLU}(\text{gate}) \cdot \text{up}$ in registers using `float4` (FP32) and `uint4` (FP16) vectorized memory loads, falling back to scalar loads for unaligned edges.
**Why:** Avoids roundtripping the intermediate `silu_out` tensor through DRAM, cutting memory traffic from $20N$ bytes to $12N$ bytes ($40\%$ savings) and achieving a measured $1.52\times$ speedup on prefill ($512 \times 11008$) and $1.61\times$ speedup on decode.

### D-015: Conditional prefill register-tiled vs decode 1D strip dispatch for rmsnorm_linear
**Date:** Oct 2026 (Pre-Module 4 Fix)
**Decision:** `rmsnorm_linear` enforces dynamic conditional dispatch: $M=1$ (decode) dispatches to the 1D row-fused strip kernel (`rmsnorm_linear_fused_direct`); $M > 1$ (prefill) dispatches to `rmsnorm` followed by `matmul_register_tiled`.
**Why:** 1D row fusion at $M=512$ destroys 2D register tiling, causing blocks to repeatedly stream columns of $W$ from memory (+2.55 GB DRAM penalty, $13.6\times$ slowdown). Register-tiled GEMM reuses $W$ across all 64 tokens in the tile, running in 0.99 ms vs 13.56 ms for 1D fused. At $M=1$, 1D fusion eliminates launch latency and avoids 2D tile quantization waste, running 43% faster than separate execution.

### D-016: FlashAttention-2 online softmax formulation and causal block skipping
**Date:** Oct 2026 (Module 4)
**Decision:** Module 4 implements FlashAttention-2 with outer loop over $Q$ blocks ($B_r = 64$) and inner loop over $K,V$ blocks ($B_c = 64$). Causal masking skips strictly lower-triangular blocks entirely. GQA maps $h_q \in [0, H_Q-1]$ to $h_{kv} = \lfloor h_q / G \rfloor$ directly in memory.
**Why:** FlashAttention-2 reduces register pressure by keeping $Q$ blocks in registers/SRAM and updating the output accumulator $O_i$ in-place via online rescaling factors ($m_{\text{new}}, \ell_{\text{new}}$). Causal block skipping cuts FLOPs by $50\%$ in autoregressive prefill. Zero memory materialization of the $S \times S$ matrix reduces DRAM traffic from $O(S^2)$ to $O(S)$.


