# Coding Conventions

> Rules, patterns, and naming conventions used across the codebase. Follow these when
> writing new code; consult them when reviewing existing code.

## Language and standard

- **C++17** — no C++20 features, no compiler extensions (`CMAKE_CXX_EXTENSIONS OFF`)
- **CUDA** — compiled by `nvcc`, targeting `sm_89` (RTX 4090, Ada Lovelace)
- **Python 3** — for tooling only (reference data generation)

## Naming conventions

| Entity | Convention | Example |
|---|---|---|
| Functions, variables | `snake_case` | `reduce_sum`, `block_size` |
| Types, classes | `PascalCase` | `Tensor`, `Storage`, `DeviceBuffer` |
| Constants | `kCamelCase` | `kBlockSize`, `kTile`, `kU` |
| Macros | `SCREAMING_CASE` | `ENGINE_CHECK`, `CUDA_CHECK`, `CUDA_CHECK_KERNEL` |
| Namespaces | `snake_case` | `engine`, `engine::cuda`, `engine::cpu` |
| Files | `snake_case` | `vector_add.cu`, `test_tensor.cpp`, `bench_harness.hpp` |
| Test suites | `snake_case` | `TEST(tensor, reshape_shares_storage)` |

## Namespace structure

```cpp
engine::          // public API: Tensor, Storage, DType, Device
engine::cpu::     // CPU reference implementations (the oracle)
engine::cuda::    // CUDA kernel launchers
```

## Error handling

- **`ENGINE_CHECK(condition, message)`** — throws `EngineError` with a descriptive
  message. Use for all host-side validation.
- **`CUDA_CHECK(cuda_call)`** — wraps any CUDA API call, throws on failure.
- **`CUDA_CHECK_KERNEL()`** — checks for launch errors after a `<<<>>>` launch.
  In debug builds or with `ENGINE_SYNC_CHECK_KERNELS=ON`, also synchronizes to catch
  in-kernel faults at the launch site rather than at the next `cudaMemcpy`.

## Kernel patterns

Every CUDA kernel launcher in `include/engine/kernels.hpp` follows these rules:

1. **Takes raw pointers and dimensions**, not `Tensor` (ADR 0003)
2. **Validates on the host** (null checks, dimension checks) — readable errors
3. **Returns early for empty inputs** — an empty grid is an error on CUDA
4. **Sizes the grid to fill the machine**: `min(blocks_needed, num_sms * 32)`
5. **Uses grid-stride loops** — grid size is independent of input size
6. **Calls `CUDA_CHECK_KERNEL()`** after every launch
7. **Uses `__restrict__`** on all pointer parameters — free performance

## Block size convention

- `kBlockSize = 256` is the standard starting point for all kernels
- Multiple of the 32-wide warp — no wasted threads
- Small enough for good occupancy, large enough to amortize per-block costs
- Treat as a tunable, not a constant. Measure before changing.

## Tolerance conventions

- Every tolerance carries its arithmetic derivation in a comment
- `vector_add`: bit-exact (0 rtol, 0 atol) — single float add, IEEE-754 guarantees
- `reduce_sum`: tree-reduction bound `2 · u · (log₂(n) + 1) · Σ|x|`
- `softmax_rows`: `u · |x − max|` (exp argument error → output relative error)
- `rmsnorm`: same as softmax, one reduction
- `matmul`: `K · u · Σ|aₖbₖ|` — K-length dot product, atol = 5e-5, rtol = 1e-4

## Test conventions

- **`TEST(suite, name)`** — real test, must pass. Promoted from `TEST_PENDING`.
- **`TEST_PENDING(suite, name)`** — unimplemented. Runs but failures report as pending.
  The test runner prints `[PENDING-PASS]` when it starts passing — time to promote.
- **`SKIP_TEST(reason)`** — prerequisite missing (no GPU, no reference data).
  Not a failure.
- **`LOAD_GOLDEN(g, stem)`** — loads reference data from `tests/golden/<stem>.bin`.
  Skips if the file is missing.
- A filter that matches zero tests exits with code 2 (not 0) — a renamed suite cannot
  leave behind a green ctest entry that runs nothing.

## Benchmark conventions

- CUDA events, not `std::chrono` (a launch returns in 3–10 µs before the kernel runs)
- Discard warmup iterations
- Report **median** and **min**, not mean (GPU timing noise is one-sided)
- Print a **spread** column: `(max − min) / median`. Flag any row above 10%.
- **Ideal traffic**: each input read once, each output written once. A naive matmul's
  re-reads appear as low achieved bandwidth, not a fictitious figure above peak.
- Every table carries a provenance footer: device, theoretical peak bandwidth, CUDA
  arch, build type, timestamp.

## File organization

- **Headers in `include/engine/`** — public API
- **Implementation in `src/`** — C++ files
- **CUDA kernels in `kernels/`** — one file per kernel or kernel pair
- **Tests in `tests/`** — one file per test suite
- **Benchmarks in `bench/`** — one file per benchmark category
- **Reference data in `tests/golden/`** — gitignored, regenerated

## Git conventions

- One exercise per commit, with test result and benchmark delta in the message
- Never commit `tests/golden/` (regenerate from `tools/gen_reference.py`)
- Never commit build artifacts, profiler reports, or model weights
- Code changes happen on Machine A; Machine B only pulls and runs
