# 01 — Development environment

Two machines, one repository.

| | Machine A — the laptop | Machine B — the 4090 box |
|---|---|---|
| GPU | none usable for CUDA | GeForce RTX 4090, `sm_89` |
| What happens here | write C++, run the CPU test suite, run `bench_cpu_ref`, edit docs, commit | compile the kernels, run the `kernels` suite, run `bench_kernels`, profile with Nsight |
| CUDA in the build | absent — `ENGINE_CUDA_ENABLED=OFF` | present — `ENGINE_CUDA_ENABLED=ON` |
| Test suites that run | `dtype`, `golden`, `cpu_ref`, `storage`, `tensor` | all of the above, plus `kernels` |
| Access | yours | yours, exclusive |

Git is the transport. Nothing is copied by hand, no files are edited over a remote
desktop session, and no build artefacts cross between the two. The reason for that rule
is narrow and worth stating: the moment you start editing on both machines you will
eventually benchmark a kernel whose source is not the source you committed, and the
number will be unreproducible. Machine A is where code changes; Machine B pulls.

That said, exercises 2–6 are kernels, and you cannot compile a kernel on Machine A.
So the loop is not "write everything, then go test it". See
[the inner loop](#the-inner-loop) below for what to do instead.

## Why the build is CUDA-optional

Most of the hours on this project are C++ hours: the tensor library, the loader, the
tokenizer, the sampler, the CPU reference implementations, the test harness. If a
missing `nvcc` broke the configure step, Machine A would stop being a development
machine the moment the first kernel landed, and you would be doing all your work over
a remote session on a box you have to keep awake.

So `nvcc` is detected, never required. Everything that touches the GPU sits behind
`#if ENGINE_HAS_CUDA`, and `.cu` files are added to the build only when there is a
compiler for them. The full reasoning is in
[ADR 0001](adr/0001-cuda-optional-build.md).

The practical consequence you should internalise: **a green test run on Machine A
proves nothing about your kernels.** It proves the CPU oracle still matches the
reference data. The `kernels` suite is simply absent there — not passing, absent.

## Machine A setup

You need a C++17 compiler, CMake ≥ 3.20, and Python 3 for the reference data.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The configure summary should end with:

```
  CUDA             : DISABLED  <-- CPU reference path only
```

That is the correct and expected state here. If it says ENABLED on the laptop,
something unexpected is installed and the CPU-only path is no longer being exercised —
which matters, because that path is what CI checks.

Then generate the reference data once:

```bash
python3 tools/gen_reference.py            # or: cmake --build build --target reference_data
```

This writes ~28 MiB into `tests/golden/`, which is gitignored. It is generated data:
regenerate it, never commit it. Until you do, the `cpu_ref` and `kernels` suites skip
with the generator command in the skip message rather than failing — "you have not
generated the reference data yet" is not a broken kernel, and a harness that reported
it as one would train you to ignore red builds.

`gen_reference.py` prefers PyTorch and falls back to NumPy (`--backend numpy`). Either
is fine; both compute in float64 and the tolerances in the tests are derived against
float64 truth.

## Machine B setup (Windows + RTX 4090)

Install, in this order:

1. **Visual Studio 2022 Build Tools**, with the "Desktop development with C++"
   workload. This is not optional and not replaceable by MinGW: on Windows, `nvcc`
   uses `cl.exe` as its host compiler. There is no CUDA-on-Windows path that does not
   go through MSVC.
2. **CUDA Toolkit 12.x.** Must be ≥ 11.8, because that is the first release that can
   emit `sm_89` at all. If `nvcc --list-gpu-arch` does not mention `compute_89`, the
   toolkit is too old and the build will warn you about exactly this at configure time.
3. **CMake ≥ 3.20** and **Ninja** (Ninja is bundled with the VS Build Tools).
4. **Git**.

Then build from an **x64 Native Tools Command Prompt for VS 2022** — a plain `cmd` or
PowerShell window will not have `cl.exe` on `PATH`, and the CUDA detection will fail
with a message about no working host compiler that reads as if CUDA itself is missing.

```bat
git clone <your-repo-url> bare-metal-llm-engine
cd bare-metal-llm-engine
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

Confirm the summary:

```
  CUDA             : ENABLED (nvcc 12.x)
  CUDA arch        : 89
  Sync-check       : OFF  (ON = debuggable, benchmarks invalid)
```

If `CUDA arch` is anything other than 89, you are compiling for the wrong GPU and the
binary will not run. `ENGINE_CUDA_ARCH` is a cache variable — pass
`-DENGINE_CUDA_ARCH=89` and reconfigure.

`scripts/build.ps1` wraps the above, and `scripts/gpu-run.sh` runs the whole
pull → configure → build → generate → test → benchmark sequence in one go.

### If you would rather use WSL2

It works, and the Linux toolchain is more pleasant. Two caveats, both real:

- You need a recent Windows NVIDIA driver; you do **not** install a driver inside
  WSL2, only the toolkit. Installing a Linux driver inside WSL2 breaks the passthrough.
- `nvidia-smi --lock-gpu-clocks` must be run from **Windows**, not from inside WSL2.
  Clock control is a driver-level operation on the host. Benchmarking from WSL2 with
  unlocked clocks gives you the 10–20% boost-related noise described below.

Native Windows is the documented path because clock locking and Nsight both live there
anyway, and the whole point of Machine B is measurement.

## The inner loop

Kernels are written on Machine A and compiled on Machine B, which sounds slow and is
not, provided you use the right feedback signal at each stage.

**On Machine A, before you push:**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
```

This catches the mistakes that are not CUDA mistakes, which is most of them: a
signature that no longer matches `cpu_ref.hpp`, a broken CPU oracle, a header that
does not compile. It cannot catch anything inside a `__global__` function.

You can also syntax-check a kernel without a GPU, which catches typos and type errors
in the host half of a `.cu` file long before you push. There is no toolkit on Machine A,
so this is not a substitute for compiling — but it is a two-second check.

**On Machine B:**

```bash
git pull
cmake --build build -j
ctest --test-dir build --output-on-failure -R kernels
```

Then, once the tests are green and only then:

```bash
build/bin/bench_kernels
```

Benchmarking a kernel that is not yet correct is wasted time in the most literal sense:
you will optimise the wrong arithmetic. The order is always correctness, then speed.

**Commit granularity.** One exercise per commit, with the test result in the message.
By June 2027 the git log is the primary evidence of how the project progressed, and
"implemented tiled matmul, 512³ 4.1 ms → 0.9 ms, kernels suite green" is worth
considerably more at a mentor checkpoint than "wip".

## Before you record any benchmark number

The 4090 boosts and throttles continuously. Back-to-back runs of an unchanged kernel
can differ by 10–20% for that reason alone — larger than most of the optimisations you
will be trying to measure. Since you have exclusive access to the machine, lock the
clocks:

```bat
nvidia-smi -q -d SUPPORTED_CLOCKS    :: see what is available
nvidia-smi -pm 1                     :: persistence mode on
nvidia-smi -lgc 2520                 :: lock graphics clock, MHz
:: ... run benchmarks ...
nvidia-smi -rgc                      :: reset afterwards
```

Run these from an Administrator prompt. Pick a clock the card can hold indefinitely
rather than its maximum boost: the goal is a number you can reproduce next week, not
the largest number available today.

Record the locked clock next to the results in
[the lab notebook](lab-notebook.md). Two benchmark runs at different clocks are not
comparable, and six months from now you will not remember which was which.

`bench_kernels` flags any row whose `(max − min) / median` exceeded 10% with `(!)`.
If you see those, the measurement is telling you it does not trust itself — find out
why before writing the number down.

## Common failure modes

**`CUDA arch 89 is NOT supported by this nvcc`** — the toolkit predates 11.8. Upgrade
it. Do not "fix" this by passing a lower arch: `sm_86` code runs on a 4090 through PTX
JIT, so it will appear to work while giving you neither Ada's instruction set nor a
representative benchmark.

**Kernel tests fail but you cannot tell which launch caused it.** In a release build,
`CUDA_CHECK_KERNEL()` only checks asynchronously, so an in-kernel fault surfaces at the
next synchronising call — usually a `cudaMemcpy` in `download()`, several lines away
from the kernel that actually faulted. Configure a second build directory with the
checks forced on:

```bash
cmake -B build-sync -DENGINE_SYNC_CHECK_KERNELS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sync -j && ctest --test-dir build-sync -R kernels --output-on-failure
```

Now the error names the launch. Keep it in a *separate* build directory, because that
option makes every launch in the library synchronise and turns every benchmark number
into a measurement of `cudaDeviceSynchronize()`.

**`no CUDA device visible -- check nvidia-smi`** from a binary that built fine.
The toolkit is installed but the driver sees no GPU. On Windows this is usually a
remote-desktop session detaching the display driver; on WSL2 it is usually a driver
that predates GPU passthrough.

**Benchmark numbers an order of magnitude worse than expected on small kernels.**
Check the `Build:` line in the table footer. If it says `DEBUG`, the harness is
measuring per-launch synchronisation. Reconfigure with `RelWithDebInfo`.

**The `kernels` ctest entry does not exist on Machine B.** CUDA was not detected. Read
the configure summary rather than guessing — it prints exactly why.
