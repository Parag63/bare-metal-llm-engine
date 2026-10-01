# Add a New Kernel

> **When to use:** When starting a new exercise in the kernel ladder, or adding a new
> operation to the engine beyond the original six exercises.
>
> **Prerequisites:** Previous exercises completed (the kernel ladder builds sequentially)
>
> **Time estimate:** Variable — from 2 hours (rmsnorm, reuses reduction) to 2 days
> (matmul_tiled, new concepts)

## Steps

### 1. Read the stub file first

Every kernel in `kernels/` has a comprehensive header comment with:
- **Spec** — what the kernel computes
- **Algorithm sketch** — suggested decomposition
- **Pitfalls** — common bugs specific to this kernel
- **Acceptance criteria** — the shapes, tolerances, and edge cases the test checks

**Read the entire header before writing any code.** It will save more time than it costs.

### 2. Verify the CPU oracle passes

```bash
build/bin/engine_tests --filter=cpu_ref.<kernel_name>
```

The CPU oracle (`src/cpu_ref/<kernel>_cpu.cpp`) should already pass against the golden
data. If it doesn't, fix the oracle first.

### 3. Implement the kernel

In `kernels/<kernel_name>.cu`:

1. Remove the `ENGINE_CHECK(false, "... not implemented yet ...")` line
2. Remove the `(void)stream;` line
3. Write your `__global__` kernel function(s) inside the anonymous namespace
4. Fill in the launcher function with:
   - Grid sizing: `min(blocks_needed, num_sms * 32)`
   - Kernel launch: `kernel<<<grid, kBlockSize, 0, stream>>>(args...)`
   - Error check: `CUDA_CHECK_KERNEL()`

### 4. Run the tests

```bash
# On Machine A (compile check only — no CUDA tests)
cmake --build build -j
ctest --test-dir build --output-on-failure

# On Machine B (the real test)
git push
# ... on Machine B:
git pull && cmake --build build -j
build/bin/engine_tests --filter=kernels.<kernel_name>
```

### 5. Promote the tests

When you see `[PENDING-PASS]` in the test output:

In `tests/test_kernels.cu`, change:
```cpp
TEST_PENDING(kernels, <kernel_name>_matches_reference) {
```
to:
```cpp
TEST(kernels, <kernel_name>_matches_reference) {
```

### 6. Run benchmarks and record

See [benchmark.md](benchmark.md) for the full procedure.

### 7. Write the lab notebook entry

In `docs/lab-notebook.md`, add a new week entry using the template. Include:
- What you did (implementation decisions)
- Test results (newly promoted tests)
- Prediction (written BEFORE measuring)
- Measurement (full benchmark table with provenance)
- Prediction vs measurement analysis
- What did not work

### 8. Commit

```bash
git add kernels/<kernel_name>.cu tests/test_kernels.cu docs/lab-notebook.md
git commit -m "Exercise N: <kernel_name> — passed X, pending Y"
```

## Checklist for a new kernel

- [ ] Read the stub header comment end-to-end
- [ ] CPU oracle passes (`cpu_ref` tests green)
- [ ] `__global__` kernel implemented
- [ ] Launcher validates inputs on the host
- [ ] Empty inputs return early (no empty grid launch)
- [ ] Grid sized to fill the machine, not match input
- [ ] Grid-stride loop for input-independent launch config
- [ ] `CUDA_CHECK_KERNEL()` after launch
- [ ] `__restrict__` on all pointer parameters
- [ ] Bounds checks on all loads/stores
- [ ] Tests pass on Machine B
- [ ] `TEST_PENDING` promoted to `TEST`
- [ ] Benchmark recorded with locked GPU clocks
- [ ] Lab notebook entry written with prediction-before-measurement
- [ ] Committed with descriptive message

## Adding a completely new kernel (beyond existing ladder exercises)

If you're adding a kernel that isn't one of the existing exercises:

1. **Create the kernel file:** `kernels/<name>.cu`
2. **Add the declaration** to `include/engine/kernels.hpp`
3. **Add the CPU oracle** to `src/cpu_ref/<name>_cpu.cpp` and `include/engine/cpu_ref.hpp`
4. **Add reference data** generation to `tools/gen_reference.py`
5. **Add test cases** to `tests/test_kernels.cu` (or a new test file)
6. **Add benchmark cases** to `bench/bench_kernels.cu`
7. **Register the source file** in `CMakeLists.txt` under `ENGINE_CUDA_SOURCES`
8. **Write an ADR** if the design decision is significant
