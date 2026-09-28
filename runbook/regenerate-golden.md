# Regenerate Reference Data

> **When to use:** After modifying `tools/gen_reference.py`, adding new test cases,
> or on a fresh clone (golden files are gitignored).
>
> **Prerequisites:** Python 3, PyTorch or NumPy
>
> **Time estimate:** ~2 seconds

## Steps

1. **Run the generator:**
   ```bash
   python3 tools/gen_reference.py
   ```
   This writes ~28 MiB of float64 binary files to `tests/golden/`.

2. **Alternative: use NumPy backend** (if PyTorch is not installed):
   ```bash
   python3 tools/gen_reference.py --backend numpy
   ```

3. **Verify the files exist:**
   ```bash
   ls tests/golden/
   ```
   You should see files like:
   ```
   vector_add__n1.bin
   vector_add__n31.bin
   reduce_sum__n1.bin
   softmax_rows__1x1.bin
   ...
   ```

4. **Re-run the tests to confirm:**
   ```bash
   ctest --test-dir build --output-on-failure
   ```
   The `golden` test suite should pass. No tests should report "golden file not found".

## Also available via CMake

```bash
cmake --build build --target reference_data
```

## Important notes

- **Golden files are gitignored.** They are regenerated, not committed. A stale golden
  file is worse than a missing one — missing causes a skip, stale causes a wrong failure.
- **Reproducible.** The generator uses fixed seeds, so regenerating produces identical
  files every time.
- **Both machines need their own copy.** After a fresh clone on Machine B, run the
  generator there too.

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Use `--backend numpy` instead |
| Tests skip with "golden file not found" | Run `python3 tools/gen_reference.py` |
| Tests fail after regenerating | Check if `gen_reference.py` was modified — tolerance expectations may need updating |
