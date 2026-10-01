# Profile Kernels with NVIDIA Nsight Compute (ncu)
 
> **When to use:** When diagnosing memory bandwidth bottlenecks, warp occupancy, shared memory bank conflicts, or validating kernel roofline positions.
>
> **Prerequisites:** Machine B (NVIDIA GPU), Nsight Compute (`ncu`) installed, build compiled in `RelWithDebInfo`.
>
> **Time estimate:** ~5–10 minutes

## 1. Prerequisites and Hardware Permissions

On modern NVIDIA drivers and WSL environments, non-root processes cannot read GPU hardware performance counters by default, triggering:
```
ERR_NVGPUCTRPERM: The user does not have permission to profile with target GPU.
```

### To fix on Windows / WSL:
1. Open **NVIDIA Control Panel** on the Windows host.
2. Navigate to **Developer** -> **Manage GPU Performance Counters**.
3. Select **"Allow access to the GPU performance counters to all users"**.
4. Click **Apply**.
5. Inside WSL, verify access with:
   ```bash
   ncu --query-metrics
   ```

## 2. Using the Automated Profiling Script

A convenience script `scripts/profile_kernels.sh` is provided in the repository.

```bash
# Make script executable
chmod +x scripts/profile_kernels.sh

# Profile all kernels with default full metrics (SpeedOfLight + Memory)
./scripts/profile_kernels.sh

# Profile a specific kernel by regex filter
./scripts/profile_kernels.sh --kernel "gemv"

# Set a custom output report filename
./scripts/profile_kernels.sh --kernel "matmul_tiled" --output reports/tiled_profile.ncu-rep
```

## 3. Manual NCU CLI Commands

### Basic Speed-of-Light (SOL) Overview
```bash
ncu --set speed_of_light -k "regex:gemv" ./build/bin/bench_kernels
```

### Detailed Memory Workload Analysis (DRAM, L2, L1, Shared Memory)
```bash
ncu --section MemoryWorkloadAnalysis -k "regex:gemv" ./build/bin/bench_kernels
```

### Check for Shared Memory Bank Conflicts
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    -k "regex:matmul_tiled" ./build/bin/bench_kernels
```

### Full Roofline Analysis and Report Generation
```bash
ncu --set full -o profile_report -k "regex:gemv" ./build/bin/bench_kernels
```
This generates `profile_report.ncu-rep`, which can be inspected directly in the graphical Nsight Compute UI.

## 4. Key Metrics to Inspect

| Metric | Target / Meaning | Action if suboptimal |
|---|---|---|
| **DRAM Throughput (%)** | > 85% for memory-bound kernels (`vector_add`, `reduce`, `rmsnorm`, `gemv`) | If low, check memory coalescing (use `float2`/`float4` loads) |
| **Compute (SM) Throughput (%)** | High for GEMM ($M \ge 512$), naturally low for GEMV ($M=1$) | Check instruction pipeline stalls, unroll inner loops |
| **Achieved Occupancy** | > 50% active warps per SM | If low, check register count per thread (`--ptxas-options=-v`) or shared memory footprint |
| **Shared Memory Bank Conflicts** | 0 conflicts per warp | If > 0, check stride indexing across the 32 shared memory banks |

## 5. Troubleshooting

| Problem | Solution |
|---|---|
| `ERR_NVGPUCTRPERM` | Enable GPU performance counter access for all users in Windows NVIDIA Control Panel |
| Kernel name not matched | Check exact mangled/unmangled kernel name using `ncu --list-kernels ./build/bin/bench_kernels` |
| Profiling takes several minutes | Use `-k <regex>` to isolate only the single kernel under study |
| Report file corrupted | Ensure clean exit without `Ctrl+C` interrupt |
