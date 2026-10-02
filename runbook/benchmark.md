# Run Benchmarks with Locked Clocks

> **When to use:** When recording performance numbers for the lab notebook. Unlocked
> GPU clocks fluctuate under thermal throttling, making measurements incomparable.
>
> **Prerequisites:** Machine B (RTX 4070 SUPER), admin/root access for `nvidia-smi`, tests
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

# Lock clocks to a specific frequency (example: 2475 MHz for RTX 4070 SUPER)
sudo nvidia-smi -lgc 2475

# Verify
nvidia-smi -q -d CLOCK
```

Common clock values:
- **RTX 4070 SUPER**: Base **1980 MHz**, Boost **2475 MHz**

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
# CPU reference baselines (standard table)
build/bin/bench_cpu_ref

# CUDA kernel benchmarks (standard table)
build/bin/bench_kernels

# Export structured JSON with git hash and driver provenance
build/bin/bench_kernels --json > results.json
```

### 4. Automated README Table Synchronization

Instead of manual copy-pasting, use the automated table generator to synchronize `README.md`:

```bash
# Option A: Ingest an existing JSON file and update README.md (Recommended)
python3 tools/generate_results_table.py --input results.json --update-readme

# Option B: Run benchmark directly and update README.md
python3 tools/generate_results_table.py --update-readme
```

This updates the benchmark table and provenance block in `README.md` without human error.

### 5. Generate Roofline Plot

Generate an empirical roofline visualization comparing all measured kernels against theoretical DRAM bandwidth and FP32 compute ceilings:

```bash
# Recommended (always pass --input to prevent non-interactive stdin blocking under WSL):
python3 tools/roofline_plot.py --input results.json --output docs/roofline.png
```

### 6. External Comparison with `llama-bench` (llama.cpp)

To evaluate our hand-written kernels against the industry-standard `llama.cpp` inference engine:

```bash
# 1. Clone and build llama.cpp with CUDA enabled
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# 2. Run prompt processing (prefill, M=512) and token generation (decode, M=1) benchmarks
# Example for TinyLlama-1.1B or LLaMA-2-7B GGUF models:
./build/bin/llama-bench -m models/tinyllama-1.1b-chat.Q4_K_M.gguf -p 512 -n 128 -t 1

# 3. Parameter alignment:
# - Prefill throughput (tokens/sec) corresponds to batched GEMM / attention speed (M=512)
# - Decode throughput (tokens/sec) corresponds directly to our gemv kernel throughput (M=1)
# 4. Record comparison in docs/lab-notebook.md under the active milestone
```

### 7. Record the results in the Lab Notebook

In `docs/lab-notebook.md`:
1. **Paste the benchmark table or reference the generated JSON artifact** including provenance (device, driver, clocks, commit hash).
2. **Record the locked GPU clock** next to the table.
3. **Note the build type** (RelWithDebInfo, Release, Debug).
4. **Check the spread column** — flag anything above 10% as untrustworthy.

### 8. Unlock the clocks when done

```bash
sudo nvidia-smi -rgc
```

## Windows (PowerShell — Machine B)

```powershell
# Lock clocks (as Administrator)
nvidia-smi -pm 1
nvidia-smi -lgc 2100

# Run benchmarks inside WSL
wsl ./build/bin/bench_kernels
wsl python3 tools/generate_results_table.py --update-readme
wsl python3 tools/roofline_plot.py

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
  or compute-bound against the RTX 4070 SUPER's ~70 FLOP/byte (RTX 4090 ~82 FLOP/byte) balance point

## Troubleshooting

| Problem | Solution |
|---|---|
| All rows show `(!)` | Clocks not locked, or background load on GPU |
| Benchmark takes very long | Check `ENGINE_SYNC_CHECK_KERNELS` is OFF |
| `% peak BW` above 100% | Byte count formula is wrong — check ideal traffic calculation |
| Numbers don't match lab notebook | Different clock speed, build type, or driver version |
| `generate_results_table.py` fails | Ensure `build/bin/bench_kernels` exists and is built |

