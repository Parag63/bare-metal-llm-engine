# ADR 0001 — CUDA is optional at build time

**Status:** accepted · **Date:** 2026-08-26 · **Supersedes:** an earlier plan to develop
on hosted notebook GPUs

## Context

Development happens on two machines. The laptop has no NVIDIA GPU; a separate Windows
box has an RTX 4090 with exclusive access. See
[docs/01-dev-environment.md](../01-dev-environment.md).

Most of the work in this project is not CUDA work. The tensor library, the GGUF-style
loader, the tokenizer, the sampler, the CPU reference implementations, the test harness,
the benchmark harness and the Python bindings are all ordinary C++. Only six files in
`kernels/` — later a few dozen — require `nvcc`.

The obvious build, `find_package(CUDAToolkit REQUIRED)` plus
`project(... LANGUAGES CXX CUDA)`, fails to configure on any machine without the
toolkit.

## Decision

CUDA is **detected, never required**. Concretely:

- `cmake/EngineCuda.cmake` uses `check_language(CUDA)`, which probes for a working
  `nvcc` without aborting, and sets `ENGINE_CUDA_ENABLED`. It cannot fail the configure
  step under any circumstance.
- `project()` declares `LANGUAGES CXX` only. `enable_language(CUDA)` is called at
  runtime, from inside the detection function, and only when `nvcc` was found.
- `.cu` files are appended to the source lists conditionally. On a CPU-only build they
  are not compiled, not linked, and not referenced.
- `cmake/engine_config.hpp.in` generates `ENGINE_HAS_CUDA` as `0` or `1`, so C++ code
  writes `#if ENGINE_HAS_CUDA` rather than guessing.
- `src/cuda_device.cpp` is compiled **unconditionally**. It is `#if`-guarded internally
  and reports "no CUDA device" on a CPU-only build, so callers — especially the
  benchmark harness — never need their own guards.
- The configure summary always prints which of the two states you got.

## Consequences

**Good.** The laptop is a real development environment rather than a degraded one: the
full non-GPU test suite builds and runs there, so does `bench_cpu_ref`, and CI can run
on a plain runner with no GPU and no CUDA toolkit. Adding a kernel never breaks anyone's
build.

**The cost, stated plainly.** A green test run on the laptop says nothing about the
kernels — the `kernels` suite is *absent* there, not passing. This is a genuine hazard:
it is easy to read a green CI badge as "everything works". Two mitigations are in place:
the configure summary states the CUDA status on every run, and `engine_tests` exits with
code 2 when a `--filter` matches no tests at all, so a `kernels` ctest entry that
silently runs nothing cannot report PASS.

**Also a cost.** Every GPU-touching header carries an `#if`. That is real complexity, and
it is concentrated deliberately: `check.hpp`, `device_buffer.hpp`, `cuda_device.hpp` and
the benchmark harness. Application code above that layer does not have guards.

## Why not the alternatives

**Require CUDA everywhere.** Would mean doing all C++ development over a remote session
on a machine that has to stay awake, for the sake of six files. Rejected on the grounds
that it makes the common case worse to simplify the rare one.

**Two separate CMake projects, CPU and GPU.** Duplicates the target definitions, the
warning flags and the test registration, and guarantees they drift. The single most
valuable property of the current setup is that the *same* `tests/test_kernels.cu`
compiles against the *same* `engine` library on both machines.

**Stub out the CUDA runtime on the laptop.** Tempting — a fake `cuda_runtime.h` would let
the `.cu` host code compile anywhere. But a stub that compiles and does nothing is worse
than an absence, because tests would then link, run, and pass against a no-op GPU. (A
stub of exactly this kind is used to *syntax-check* `.cu` host code during development,
which is a different thing from linking it into a test binary.)
