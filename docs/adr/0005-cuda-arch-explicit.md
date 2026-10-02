# ADR 0005 — `ENGINE_CUDA_ARCH` is explicit, and mismatches warn loudly

**Status:** accepted · **Date:** 2026-08-26 · **Applies to:** `cmake/EngineCuda.cmake`,
root `CMakeLists.txt`

## Context

`CUDA_ARCHITECTURES` decides which GPU architectures `nvcc` emits code for. Get it wrong
and one of three things happens, in increasing order of how long it takes to notice:

1. The binary refuses to run — obvious, fixed in a minute.
2. The binary runs, because the driver JIT-compiles the embedded PTX for the actual GPU.
   It works. It is also compiled for the wrong instruction set, and any benchmark from it
   is not a measurement of the hardware you think you are measuring.
3. `nvcc` is too old to know about the requested architecture and the build fails with a
   message that reads like a configuration error rather than "upgrade your toolkit".

Case 2 is the dangerous one. `sm_86` code runs fine on a 4090 through PTX JIT, and
nothing in the output says so.

CMake ≥ 3.24 offers `CUDA_ARCHITECTURES native`, which detects the GPU present at
configure time. That sounds strictly better, and for a single-machine project it is.

## Decision

`ENGINE_CUDA_ARCH` is an explicit cache variable, defaulting to `"89"` (Ada Lovelace,
RTX 4090), applied to every CUDA target in the project. `native` is documented as an
option but is not the default.

`cmake/EngineCuda.cmake` additionally runs `nvcc --list-gpu-arch` and emits a
`message(WARNING)` if the requested architecture is not one this toolkit can emit, naming
the specific cause: `sm_89` needs CUDA ≥ 11.8.

The value is compiled into the binary as `ENGINE_CUDA_ARCH_STRING` and printed in the
provenance footer of every benchmark table, alongside the device name and the build type.

## Consequences

**A wrong-GPU build is loud rather than silent.** The configure summary prints
`CUDA arch : 89` on every run. If that line is missing or different, you know before you
build, not after you have recorded a table of numbers.

**Benchmark results are self-documenting.** The arch string appears in the footer of every
table, so a result pasted into the lab notebook carries the architecture it was compiled
for. Six months later, comparing two tables, this is the difference between a comparison
and a guess.

**A toolkit too old for Ada says so, specifically.** The generic CMake failure for an
unsupported architecture does not mention version requirements. The warning here names
11.8, which is the actual answer.

**The cost.** Building on any other GPU requires passing `-DENGINE_CUDA_ARCH=<arch>`. For
this project that is a one-line change on a machine that does not exist yet, which is
cheaper than the failure mode it prevents. The root `CMakeLists.txt` lists the common
fallback values (86 for RTX 3090/A10, 80 for A100, 75 for T4/RTX 2080) with the explicit
note that a binary built for 89 will not run on any of them.

**A related trap, fixed in the same file.** `nvcc` does not compile host code itself — it
delegates to `cl.exe` on Windows and `g++`/`clang++` elsewhere. So
`-Xcompiler=-Wall` means two different things: the usual sensible warning set on GCC and
Clang, but on MSVC a synonym for `/Wall`, which enables *every* warning including
thousands from the Windows SDK and CRT headers. Since the 4090 box is a Windows machine,
that is the branch that runs when the kernels are actually compiled, and the flood would
bury every real warning about the kernels. `EngineCuda.cmake` therefore branches on
`MSVC` and passes `-Xcompiler=/W4` there, mirroring what
`engine_apply_warnings()` does for the C++ targets.

## Why not the alternatives

**`CUDA_ARCHITECTURES native`.** Correct and convenient on one machine. Rejected as the
default for two reasons: it requires CMake ≥ 3.24, above this project's 3.20 floor, and it
makes the compiled architecture depend on *which machine configured the build* rather
than on a value recorded in the repository. When a benchmark table has to be defensible
in a report, "the architecture is whatever the configuring machine happened to have" is a
worse property than a fixed number, even a wrong one — a wrong fixed number is
discoverable.

**Multiple architectures, e.g. `"75;86;89"`.** Standard practice for shipped software,
where the binary must run anywhere. It multiplies compile time by the number of
architectures and buys nothing for a project that runs on exactly one known GPU.

**Leave it unset and let CMake choose.** CMake's default depends on the toolkit version
and has changed between releases. That is precisely case 2 above: it works, and you do
not know what you measured.

---

## Addendum (October 2026): Headless CI Architecture Default

On headless CI runners lacking physical NVIDIA hardware, CMake defaults to `"86;89"` to validate compilation across Ampere and Ada Lovelace without requiring a runtime GPU at configure time.

