# GPU Workflow — Running CUDA Kernels on Machine B

> **When to use:** Every time you push new kernel code from Machine A and need to
> compile, test, and benchmark on the target GPU (RTX 4070 SUPER, RTX A4000, or RTX 4090).
>
> **Prerequisites:** Code pushed from Machine A, Machine B has CUDA toolkit + CMake
>
> **Time estimate:** ~2 minutes for pull + build + test

## The workflow

```
Machine A (laptop)     Machine B (RTX 4070 SUPER / A4000)
─────────────────      ──────────────────────────────────
1. Write kernel code
2. git add, commit
3. git push
                       4. git pull
                       5. cmake --build build -j
                       6. ctest --test-dir build -R kernels --output-on-failure
                       7. build/bin/bench_kernels (if tests pass)
```

## Steps on Machine B

1. **Pull the latest code:**
   ```bash
   git pull origin main
   ```

2. **Build (incremental):**
   ```bash
   cmake --build build -j
   ```

3. **Run kernel tests:**
   ```bash
   ctest --test-dir build -R kernels --output-on-failure
   ```
   Or for a specific kernel:
   ```bash
   build/bin/engine_tests --filter=kernels.softmax
   ```

4. **Run benchmarks** (only if tests pass!):
   ```bash
   build/bin/bench_kernels
   ```

5. **Record results** in `docs/lab-notebook.md`:
   - Paste the full benchmark table including the provenance footer
   - Record the locked GPU clock (see [benchmark.md](benchmark.md))
   - Compare against your prediction

## Important rules

> ⚠️ **Never edit code on Machine B.** Write on A, push, pull on B. A benchmark whose
> source isn't the committed source is unreproducible.

> ⚠️ **Always run tests before benchmarks.** Optimizing a wrong kernel means carefully
> tuning the wrong arithmetic.

## Using the GPU run script

```bash
# The provided script handles clock-locking and logging
./scripts/gpu-run.sh
```

This writes a timestamped log to `runs/` (gitignored). Numbers that matter get pasted
into the lab notebook by hand — deliberately, so entering the permanent record is a
decision, not a side effect.

## Troubleshooting

| Problem | Solution |
|---|---|
| `no CUDA device visible` | Run `nvidia-smi` — driver may not be loaded |
| Tests all skipped | CUDA not enabled in build — reconfigure with `ENGINE_WITH_CUDA=ON` |
| Wrong CUDA arch | Check `ENGINE_CUDA_ARCH="86;89"` (or `89` / `86`) in configure output |
| Kernel crashes without useful error | Build with `ENGINE_SYNC_CHECK_KERNELS=ON` (separate build dir!) |
