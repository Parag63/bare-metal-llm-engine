# Run the Test Suite

> **When to use:** After any code change, before committing.
>
> **Prerequisites:** Successful build (see [build.md](build.md)), reference data
> generated (`python3 tools/gen_reference.py`)
>
> **Time estimate:** <5 seconds (CPU-only), <15 seconds (with CUDA)

## Run all tests

```bash
ctest --test-dir build --output-on-failure
```

## Run a specific test suite

```bash
# Just tensor tests
build/bin/engine_tests --filter=tensor

# Just kernel tests
build/bin/engine_tests --filter=kernels

# A specific test
build/bin/engine_tests --filter=kernels.reduce_sum
```

## Understanding the output

The test harness reports four outcomes:

| Symbol | Meaning |
|---|---|
| **pass** | ✅ Verified against reference data |
| **fail** | ❌ Broken — exit code 1 |
| **pending** | 🟡 Module not written yet. Reports `[PENDING-PASS]` when it starts passing |
| **skipped** | ⏭️ Prerequisite missing (no GPU, no reference data) |

### What `[PENDING-PASS]` means

```
[PENDING-PASS] tensor.numel_is_the_product_of_shape  <-- implemented!
```

This means you've implemented the feature and the `TEST_PENDING` should be promoted to
`TEST` in the source. Change:
```cpp
TEST_PENDING(tensor, numel_is_the_product_of_shape) { ... }
```
to:
```cpp
TEST(tensor, numel_is_the_product_of_shape) { ... }
```

## Expected test suites per machine

| Suite | Machine A (no GPU) | Machine B (RTX 4090) |
|---|---|---|
| `dtype` | ✅ Runs | ✅ Runs |
| `golden` | ✅ Runs | ✅ Runs |
| `cpu_ref` | ✅ Runs | ✅ Runs |
| `storage` | ✅ Runs | ✅ Runs |
| `tensor` | ✅ Runs | ✅ Runs |
| `kernels` | ⏭️ Skipped (no CUDA) | ✅ Runs |

## Verification

A good test run looks like:
```
passed NN   failed 0   pending MM   skipped 0
```

**Never commit with `failed > 0`.** Pending is fine — it means a feature isn't
implemented yet. Failed means something is broken.

## Troubleshooting

| Problem | Solution |
|---|---|
| All kernel tests skipped | No CUDA device — expected on Machine A |
| `golden file not found` | Run `python3 tools/gen_reference.py` first |
| Filter matches zero tests | Exits with code 2 — check the filter spelling |
| Tests pass but harness says `passed 0` | A renamed suite left behind a stale ctest entry — rebuild |
