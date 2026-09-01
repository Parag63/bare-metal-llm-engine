# ADR 0002 — `Storage` and `Tensor` are separate types

**Status:** accepted · **Date:** 2026-08-26 · **Implements:** Module 1, Objective 1
(scheduled October 2026)

## Context

A tensor needs two things that have different lifetimes: a buffer of bytes on some
device, and an interpretation of those bytes (shape, strides, dtype, offset). The
simplest design fuses them — one class that owns its allocation and describes its own
shape.

Three requirements later in this project are incompatible with that:

- **KV-cache slices** (Feb 2027). Each generation step reads `cache[:, :pos, :]`. If a
  slice copies, the cache costs more than recomputing and the entire optimisation is
  pointless.
- **Memory-mapped GGUF weights** (Mar 2027). Roughly 50 weight tensors must alias one
  `mmap` region. If each owns its bytes, peak memory doubles — on a 4 GB quantized model
  that is the difference between fitting in VRAM and not.
- **Reshapes in attention.** `[B, S, H·D] → [B, S, H, D]` must be free; it happens
  several times per layer per token.

All three are *views*: different shape metadata over the same bytes.

## Decision

Two types.

```
Storage  -- reference-counted ownership of a flat byte buffer on ONE device.
            Non-copyable, non-movable; shared exclusively through shared_ptr.
Tensor   -- a view: shared_ptr<Storage> + shape + strides + byte offset + dtype.
```

Many `Tensor`s may share one `Storage`. The buffer lives exactly as long as the last
view referring to it. This is the design PyTorch uses, for the same reasons.

`Storage` is deliberately move-deleted as well as copy-deleted: it is only ever handled
through `shared_ptr`, and allowing a move would create a second path to the same pointer
with different ownership semantics.

## Consequences

**Good.** `transpose`, `slice`, `reshape` (when compatible) and `view` are O(1) metadata
edits that touch no data. The mmap loader and the KV cache become straightforward rather
than special cases. Lifetime is automatic: a view keeps its storage alive.

**The cost.** Every element access goes through one extra indirection, and every `Tensor`
carries a `shared_ptr` — atomic refcount traffic on copy. This is irrelevant for kernel
launches, where a pointer is extracted once per launch and the kernel then runs for
microseconds. It would matter in a scalar element-access loop, which is not something
this engine does on a hot path.

**The real cost, and it is a genuine one:** strides make data potentially
non-contiguous, and most CUDA kernels cannot handle arbitrary strides. So the codebase
needs `is_contiguous()` and `contiguous()`, kernels must assert contiguity, and there is
now a class of bug where a transposed view is passed to a kernel that silently assumes
row-major. Making the kernel API take raw pointers (see
[ADR 0003](0003-raw-pointer-kernel-api.md)) puts that check at a single explicit
boundary rather than scattering it.

## Why not the alternatives

**One class that owns its data.** Simpler for about two months. Retrofitting views onto
it later means changing every construction site, every function signature that takes a
tensor by value, and every ownership assumption in the loader — a refactor across the
whole codebase at the exact moment the KV cache is being written. Paying the indirection
cost now is cheaper than paying that later.

**Views as a separate `TensorView` type.** Then every function must be written twice, or
templated, or take a base class — and the question "does this function copy?" becomes a
signature detail rather than an obvious property. Making *every* `Tensor` a view removes
the distinction entirely: nothing copies unless you call `contiguous()` or `clone()`.

**Raw pointers with manual lifetime management, llama.cpp-style.** Defensible in C, and
llama.cpp does roughly this with a context-owned arena. But `shared_ptr` costs nothing
measurable at this granularity, and a use-after-free in a KV cache is a debugging session
that produces plausible-looking wrong logits rather than a crash. Not a trade worth
making for a solo project on a deadline.
