# ADR 0003 — Kernel launchers take raw pointers, not `Tensor`

**Status:** accepted · **Date:** 2026-08-26 · **Applies to:** `include/engine/kernels.hpp`

## Context

The host-side launch API could take the project's own `Tensor` type, which knows its
shape, dtype, device and contiguity, or it could take raw device pointers plus explicit
`std::int64_t` dimensions.

Tensor-typed signatures are more expressive and catch more mistakes at compile time.
They also create a dependency: nothing that takes a `Tensor` can be compiled, tested or
benchmarked until `Tensor` works.

## Decision

Every function in `engine/kernels.hpp` takes raw device pointers, explicit dimensions,
and a `cudaStream_t` defaulting to the legacy stream `0`:

```cpp
void matmul_tiled(const float* A, const float* B, float* C,
                  std::int64_t M, std::int64_t N, std::int64_t K,
                  cudaStream_t stream = 0);
```

Tensor-aware overloads will be added later as thin wrappers that unpack `.data()` and
`.shape()`, assert contiguity and device, and forward to these.

Each launcher mirrors its CPU twin in `engine/cpu_ref.hpp` with the same name and
argument order.

## Consequences

**The scheduling one, which is the immediate reason.** The kernel ladder is August–
December 2026 work; `Tensor` is scheduled for October. With Tensor-typed signatures,
exercise 2 could not be compiled, let alone tested, until Module 1 was finished — so the
two modules would have to be developed in series with the harder one blocked behind the
easier one. Raw pointers break the dependency, and the exercises can proceed now.

**It is also the correct layering, independent of scheduling.** cuBLAS, cuDNN and
llama.cpp's ggml all expose pointer-plus-dimensions at the kernel boundary and keep the
tensor abstraction strictly above it. A kernel should not know what a `Tensor` is: it
needs an address, a count, and a stride convention.

**It makes the test harness an A/B comparison.** Because the CUDA and CPU signatures
match, `tests/test_kernels.cu` and `tests/test_cpu_ref.cpp` can share the same golden
files, the same case tables and the same `CHECK_CASE` macro. The kernel tests are
literally the CPU tests with a different callee.

**The cost, which is real.** A host pointer passed where a device pointer is expected
compiles cleanly and then faults at runtime — or, worse, silently reads garbage. Nothing
in the type system prevents it. Two mitigations:

- `engine::DeviceBuffer<T>` is the intended way to hold device memory, and `.get()` is
  the only way to obtain the pointer, so the mistake requires effort.
- Every launcher validates its arguments with `ENGINE_CHECK` before doing anything else:
  non-negative dimensions, non-null pointers. `kernels.launchers_reject_negative_dimensions`
  and `kernels.launchers_reject_null_pointers` assert this for all six.

That validation ordering turned out to matter more than expected. Because `ENGINE_CHECK`
runs *before* the "not implemented" sentinel in each stub, three whole test cases — 26
assertions covering negative dimensions, null pointers and empty-work no-ops — are real
passing tests today, before a single kernel body exists. They also pin the contract so it
survives the removal of the stubs.

**Dimensions are `std::int64_t`, not `std::size_t`.** Signed, so that a subtraction
producing a negative value is detectably negative rather than wrapping to an enormous
positive count. The `ENGINE_CHECK(n >= 0, ...)` in each launcher is only meaningful with
a signed type.

## Why not the alternatives

**Tensor-typed from the start.** Blocks the entire kernel ladder behind Module 1, and
couples the lowest layer of the system to the type most likely to change during
implementation.

**Both, from the start.** Doubles the surface to be tested and documented before there is
any evidence about what the wrapper should assert. The wrapper is cheap to add once
`Tensor` exists and once the kernels have settled; adding it now would be designing
against an unwritten class.

**A lightweight `TensorRef` struct (pointer + dims) at the boundary.** Genuinely
attractive, and worth revisiting if the argument lists grow. Rejected for now because it
would be a third type that means "almost a tensor", and because the CPU/CUDA signature
symmetry — which is what makes the three-tier test harness trivial — is easier to
maintain when both sides take plain scalars.
