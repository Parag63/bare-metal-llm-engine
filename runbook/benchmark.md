# Run Benchmarks with Locked Clocks

> **When to use:** When recording performance numbers for the lab notebook. Unlocked
> GPU clocks fluctuate under thermal throttling, making measurements incomparable.
>
> **Prerequisites:** Machine B (RTX 4090), admin/root access for `nvidia-smi`, tests
> passing for the kernel you want to benchmark.
>
> **Time estimate:** ~5 minutes including clock setup

## Why clock locking matters

The GPU dynamically adjusts clock speeds based on temperature and power. Two runs at
different clocks are **not comparable**, and you won't remember which was which. If
you didn't lock the clocks, write that down — it's the honest thing, and it explains
the spread.

## Steps

### 1. Lock the GPU clocks

```bash
# Enable persistence mode (survives between nvidia-smi calls)
sudo nvidia-smi -pm 1

# Lock clocks to a specific frequency (example: 2100 MHz for RTX 4090)
sudo nvidia-smi -lgc 2100

# Verify
nvidia-smi -q -d CLOCK
```

Common clock values for RTX 4090:
- **2100 MHz** — a good stable point below max boost
- **2520 MHz** — max boost (may thermal throttle under sustained load)

### 2. Build in release mode

```bash
# For benchmark accuracy, use RelWithDebInfo (the default)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

> ⚠️ **Never benchmark with `ENGINE_SYNC_CHECK_KERNELS=ON`.** Use a separate build
> directory for debugging.

### 3. Run the benchmarks

```bash
# CPU reference baselines
build/bin/bench_cpu_ref

# CUDA kernel benchmarks
build/bin/bench_kernels
```

### 4. Record the results

In `docs/lab-notebook.md`:
1. **Paste the entire benchmark table** including the provenance footer (device,
   theoretical peak bandwidth, CUDA arch, build type, timestamp)
2. **Record the locked GPU clock** next to the table
3. **Note the build type** (RelWithDebInfo, Release, Debug)
4. **Check the spread column** — flag anything above 10% as untrustworthy

### 5. Unlock the clocks when done

```bash
sudo nvidia-smi -rgc
```

## Windows (PowerShell — Machine B)

```powershell
# Lock clocks (as Administrator)
nvidia-smi -pm 1
nvidia-smi -lgc 2100

# Run benchmarks
.\build\bin\bench_kernels.exe

# Unlock when done
nvidia-smi -rgc
```

## Reading the benchmark table

```
| kernel | size | median (ms) | min (ms) | spread | GFLOP/s | GB/s | % peak BW | AI (FLOP/B) |
```

- **median** — the number to quote. Less sensitive to outliers than mean.
- **min** — the best the kernel achieved (useful for roofline analysis)
- **spread** — `(max − min) / median`. Above 10% = flag with `(!)` and explain why
- **% peak BW** — how close to the hardware's theoretical memory bandwidth
- **AI (FLOP/B)** — arithmetic intensity. Determines whether the kernel is memory-bound
  or compute-bound against the RTX 4090's ~82 FLOP/byte balance point

## Troubleshooting

| Problem | Solution |
|---|---|
| All rows show `(!)` | Clocks not locked, or background load on GPU |
| Benchmark takes very long | Check `ENGINE_SYNC_CHECK_KERNELS` is OFF |
| `% peak BW` above 100% | Byte count formula is wrong — check ideal traffic calculation |
| Numbers don't match lab notebook | Different clock speed, build type, or driver version |
