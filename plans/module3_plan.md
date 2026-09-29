# Module 3 — Kernel Fusion: Implementation Plan

> **Goal:** Eliminate redundant global-memory round-trips by fusing operations that the
> transformer executes back-to-back. Every transformer layer in LLaMA-7B does
> `RMSNorm → Linear` (pre-attention) and `RMSNorm → Linear` (pre-FFN) — that is 64
> kernel launches per token where 32 would do, plus 64 extra reads and writes of the
> full hidden state. Fusion fixes both.

## Why This Matters (The Quantitative Argument)

At `D = 4096` (LLaMA-7B hidden dim), one RMSNorm followed by one matmul launch means:
- **RMSNorm writes** `rows × 4096 × 4B` to DRAM
- **Matmul reads** it right back

For a single-token decode (`rows = 1`), that's 16 KiB — too small to matter. But for
prefill with `rows = 512`, it's **8 MiB of traffic that fusion eliminates entirely**, plus
the kernel launch overhead (~5 µs per launch × 64 launches = 320 µs/token of pure overhead).

The deeper lesson: fusion doesn't change the arithmetic, it changes the *memory schedule*.
That's the same idea as tiling (Module 2) and FlashAttention (Module 4+), so this module
is the bridge between "understanding the memory hierarchy" and "using it at the algorithm
level."

---

## Exercise Plan

### Exercise 7 — `rmsnorm_linear`: Fused RMSNorm + Matrix Multiply

**What it computes:**
```
    temp[r][c] = in[r][c] / sqrt(mean(in[r][:]²) + eps) * weight[c]    (RMSNorm)
    out[r][n]  = Σ_k temp[r][k] * W[k][n]                              (Linear)
```

But `temp` is never materialized in global memory — each block computes the normalized
row in shared memory / registers and immediately consumes it for the matmul accumulation.

**This is the pre-attention pattern:** `hidden → RMSNorm → QKV projection` in every
LLaMA layer. Fusing it means the 4096-wide hidden state is read once instead of twice.

**Decomposition (skeleton + guided):**

Each block computes a `1 × TILE_N` strip of the output for one row:
1. **Load row into shared memory** (cooperative, block-strided)
2. **Block-reduce** sum of squares → compute `scale = rsqrtf(mean_sq + eps)`
3. **Normalize in-place in shared memory**: `smem[c] *= scale * rms_weight[c]`
4. **Tiled matmul accumulation**: march over K in tiles, reading the normalized row
   from shared memory (already there!) and loading tiles of W from global memory.
5. **Write** the `1 × TILE_N` output strip to global memory.

The key insight: step 3's output feeds step 4's input *without a global-memory round-trip*.

### Exercise 8 — `residual_rmsnorm`: Fused Residual Add + RMSNorm

**What it computes:**
```
    sum[r][c] = x[r][c] + residual[r][c]          (residual connection)
    out[r][c] = sum[r][c] / sqrt(mean(sum[:]²) + eps) * weight[c]   (RMSNorm)
```

Again, `sum` is never written to global memory as a separate step — the normalized output
and the raw sum are written in a single pass.

**This is the post-attention / post-FFN pattern:** after every sub-layer in a transformer,
the residual connection adds the sub-layer output to the input, and then RMSNorm normalizes
for the next sub-layer. Without fusion, that's: write residual sum → read it back for
RMSNorm. Fusion eliminates that entire intermediate.

**Decomposition:** Almost identical to existing `rmsnorm.cu`:
1. Each thread loads `x[c] + residual[c]` and accumulates sum of squares.
2. Block-reduce, compute scale.
3. Write `(x[c] + residual[c]) * scale * weight[c]` to `norm_out`.
4. **Also** write `x[c] + residual[c]` to `sum_out` (the residual stream needs it
   for the *next* residual connection). This is the only extra global write.

---

## File-by-File Deliverables

Following the [add-kernel runbook](file:///c:/Users/pd207/AppData/Local/Claude-3p/local-agent-mode-sessions/388d3b34/00000000/e4c996d6/outputs/bare-metal-llm-engine/runbook/add-kernel.md):

| Step | File | Action |
|------|------|--------|
| 1 | `include/engine/kernels.hpp` | Add `rmsnorm_linear()` and `residual_rmsnorm()` declarations |
| 2 | `include/engine/cpu_ref.hpp` | Add matching CPU oracle declarations |
| 3 | `src/cpu_ref/rmsnorm_linear_cpu.cpp` | CPU oracle: RMSNorm then matmul, scalar loops |
| 4 | `src/cpu_ref/residual_rmsnorm_cpu.cpp` | CPU oracle: add + RMSNorm, scalar loops |
| 5 | `kernels/rmsnorm_linear.cu` | CUDA fused kernel (Exercise 7) |
| 6 | `kernels/residual_rmsnorm.cu` | CUDA fused kernel (Exercise 8) |
| 7 | `tools/gen_reference.py` | Add golden data for both fused ops |
| 8 | `tests/test_kernels.cu` | Add test cases for both kernels |
| 9 | `bench/bench_kernels.cu` | Add benchmark: fused vs separate |
| 10 | `CMakeLists.txt` | Register new `.cu` and `.cpp` source files |
| 11 | `docs/adr/0006-kernel-fusion-strategy.md` | ADR: why these fusions, why not others |
| 12 | `brain/STATUS.md` | Update module status |

## Kernel Signatures

```cpp
// --- kernels.hpp ---

/// Exercise 7. Fused RMSNorm + Linear projection.
/// out[M x N] = RMSNorm(in[M x K], rms_weight[K], eps) * W[K x N]
/// Eliminates the intermediate M×K write/read between RMSNorm and matmul.
void rmsnorm_linear(const float* in, const float* rms_weight,
                    const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K,
                    float eps, cudaStream_t stream = 0);

/// Exercise 8. Fused Residual-Add + RMSNorm.
/// norm_out[r][c] = RMSNorm(x[r][c] + residual[r][c], weight, eps)
/// sum_out[r][c]  = x[r][c] + residual[r][c]   (for the next residual connection)
void residual_rmsnorm(const float* x, const float* residual,
                      const float* weight, float* norm_out, float* sum_out,
                      std::int64_t rows, std::int64_t cols,
                      float eps, cudaStream_t stream = 0);
```

## CPU Oracle Signatures (matching)

```cpp
// --- cpu_ref.hpp ---

/// Fused RMSNorm + matmul. Same result as rmsnorm() followed by matmul().
void rmsnorm_linear(const float* in, const float* rms_weight,
                    const float* W, float* out,
                    std::int64_t M, std::int64_t N, std::int64_t K, float eps);

/// Fused residual-add + RMSNorm. Writes both the normalized output and the
/// pre-normalization sum (needed by the next residual connection).
void residual_rmsnorm(const float* x, const float* residual,
                      const float* weight, float* norm_out, float* sum_out,
                      std::int64_t rows, std::int64_t cols, float eps);
```

## Test Shapes

| Kernel | Shapes | Rationale |
|--------|--------|-----------|
| `rmsnorm_linear` | `(1, 4096, 4096)` | Single-token decode, LLaMA-7B dims |
| | `(32, 4096, 4096)` | Small batch |
| | `(1, 4096, 12288)` | QKV projection (3 × 4096) |
| | `(17, 127, 31)` | Edge-case: nothing divides anything |
| `residual_rmsnorm` | `(1, 4096)` | Single token |
| | `(128, 4096)` | Prefill batch |
| | `(32, 127)` | Awkward width |

## Tolerances

- **`rmsnorm_linear`**: Same bound as `rmsnorm` × `matmul` composed:
  `atol = 5e-5, rtol = 1e-4` (dominated by the K-length dot product).
- **`residual_rmsnorm`**: Same as `rmsnorm` (`rtol = 1e-5, atol = 1e-6`),
  because the add is exact (two float32 inputs, one float32 add = correctly rounded).

## Benchmark Strategy

The headline measurement: **fused vs. separate**, at the shapes that matter.

```
| kernel              | shape           | separate (µs) | fused (µs) | speedup | saved traffic |
|---------------------|-----------------|---------------|------------|---------|---------------|
| rmsnorm + matmul    | 1 × 4096→4096  | ???           | ???        | ???     | 16 KiB        |
| rmsnorm + matmul    | 512 × 4096→4096| ???           | ???        | ???     | 8 MiB         |
| residual + rmsnorm  | 512 × 4096     | ???           | ???        | ???     | 8 MiB         |
```

The speedup should be modest at small sizes (launch overhead dominates) and significant
at prefill sizes (the eliminated traffic is real).

## Implementation Order

> [!IMPORTANT]
> **Start with Exercise 8 (`residual_rmsnorm`)**. It's structurally identical to the
> existing `rmsnorm.cu` with one extra input read and one extra output write — minimal
> new concepts. Exercise 7 is harder because it combines a row reduction with a tiled
> matmul, which is new territory.

1. **Exercise 8 first** (residual_rmsnorm) — ~2 hours
   - CPU oracle → golden data → kernel → tests → promote → benchmark
2. **Exercise 7 second** (rmsnorm_linear) — ~4-6 hours
   - CPU oracle → golden data → kernel → tests → promote → benchmark
3. **ADR + lab notebook + STATUS update**
4. **Commit**: one per exercise, with test results in the message

---

> [!TIP]
> **Shall I proceed with implementing this plan?** I'll start with the infrastructure
> (CPU oracles, golden data generation, test stubs, CMake registration) for both exercises,
> then implement Exercise 8 (`residual_rmsnorm`) first as the simpler kernel.
