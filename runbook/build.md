# Build the Project

> **When to use:** First-time setup, after pulling changes, or when CMake configuration
> changes (new source files, new options, toolchain updates).
>
> **Prerequisites:** C++17 compiler, CMake ≥ 3.20, Python 3 (for reference data)
>
> **Time estimate:** ~1 minute (clean build), ~10 seconds (incremental)

## Machine A (Laptop — CPU only)

### Steps

1. **Configure** (only needed once, or when CMakeLists.txt changes):
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   ```
   The configure summary should show:
   ```
   CUDA             : DISABLED  <-- CPU reference path only
   ```

2. **Build:**
   ```bash
   cmake --build build -j
   ```

3. **Generate reference data** (only needed once, or after changing `gen_reference.py`):
   ```bash
   python3 tools/gen_reference.py
   ```
   This writes ~28 MiB of float64 golden files to `tests/golden/`.

## Machine B (RTX 4070 SUPER — with CUDA)

### Steps

1. **Pull latest code:**
   ```bash
   git pull
   ```

2. **Configure:**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   ```
   The configure summary should show:
   ```
   CUDA             : ENABLED (nvcc 12.x)
   CUDA arch        : 89
   ```

3. **Build:**
   ```bash
   cmake --build build -j
   ```

4. **Generate reference data** (if not already present):
   ```bash
   python3 tools/gen_reference.py
   ```

## Build for debugging kernel faults

If a kernel test fails in a way you cannot localize, use a separate debug build with
synchronous kernel checks:

```bash
cmake -B build-sync -DENGINE_SYNC_CHECK_KERNELS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sync -j
```

> ⚠️ **Never benchmark with this build.** Per-launch synchronization makes every timing
> number meaningless. Use a separate build directory.

## Verification

After building, run `cmake --build build --target help` to see available targets.
Key targets: `engine_tests`, `bench_cpu_ref`, `bench_kernels`, `reference_data`.

## Troubleshooting

| Problem | Solution |
|---|---|
| `nvcc not found` | Machine A: expected — CUDA is optional. Machine B: add CUDA toolkit to PATH |
| `CMAKE_CUDA_COMPILER not set` | Set `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` |
| Warnings from Windows SDK | Expected with MSVC — `EngineWarnings.cmake` handles `/W4` vs `-Wall` |
| `engine_config.hpp` not found | Run `cmake -B build` (configure step) — the header is generated |

## PowerShell (Windows)

```powershell
# Machine B is Windows — use the provided build script
.\scripts\build.ps1
```
