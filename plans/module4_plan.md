# Module 4 Implementation Plan: Attention (FlashAttention-2, GQA, RoPE, FlashDecoding)

**Written:** 2026-11-11 | **Base:** commit `4cdf216` plus the pending pre-Module 4 working tree | **GPU:** RTX 4070 SUPER only (second GPU deferred) | **Precision:** FP16 storage, FP32 accumulation and softmax statistics (ADR-0011) | **Roadmap window:** Oct to Nov 2026 (realistic finish: first days of December)

Priority tags: **[Must]** core to the result, **[Should]** strengthens it, **[Stretch]** cut first.

---

## 1. Goal

Module 4 should answer three measurable questions, not just ship kernels:

1. How much memory and time does flash tiling save over naive attention, and at what sequence length does it pay off?
2. How close can decode attention get to the memory-bandwidth ceiling, and what does split-KV (FlashDecoding) buy?
3. What does grouped-query attention (GQA) save in KV traffic, measured, not assumed?

**In scope:** naive FP16 baseline, FlashAttention-2 prefill (causal, GQA, position offset), decode attention with split-KV, RoPE (both pair conventions), CPU oracles, Tier 1 golden data, tests, benchmarks, ADRs, notebook entries.
**Out of scope (later modules):** KV-cache management and paging (Module 6), quantized KV, batching, backward pass.

## 2. Model targets

| Model | Layers | Hidden | Q heads | KV heads | Head dim | Context |
|---|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | 22 | 2048 | 32 | 4 | 64 | 2048 |
| Llama-2-7B (headline target) | 32 | 4096 | 32 | 32 | 128 | 4096 |

Support head dim 64 and 128 (templated), and any `n_heads % n_kv_heads == 0`. Reject anything else with `EngineError`.

Useful numbers for predictions (write your own predictions before measuring):
- Prefill attention FLOPs: `4 * H * D * S^2` non-causal, about half that for causal. TinyLlama at S=2048: about 34.4 GFLOP non-causal, 17.2 GFLOP causal.
- Naive FP32 score matrix at S=2048, 32 heads: 512 MiB. Flash needs none of it.
- TinyLlama KV per layer at L=2048 (FP16): 2 MiB; all 22 layers: 44 MiB, roughly the size of the 48 MB L2.

## 3. Decisions to record (ADR-0012 and ADR-0013)

Write these down before coding. Each should name the alternative you rejected.

- [ ] **Layouts.** Q and output token-major `[S, H_q, D]` (what the projection GEMM produces). K and V head-major `[H_kv, S_max, D]` (contiguous per-head streaming for decode). Kernels take explicit strides, so Module 6 can change layouts without touching kernels (ADR-0003 raw-pointer style).
- [ ] **Signature sketch** (final form in ADR-0012): `attention_prefill(q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim, q_pos_offset, causal, scale, strides..., stream)` and `attention_decode(q, k, v, out, kv_len, ..., stream)`. `q_pos_offset` lets query `i` sit at absolute position `offset + i`, which chunked prefill and cached continuation both need.
- [ ] **Contracts.** `s_q == 0` is a no-op; `s_kv == 0` writes zeros; every causal row has at least one valid key; null pointers and bad head ratios throw. (Same spirit as your `K == 0` contract.)
- [ ] **RoPE convention.** Implement both rotate-half (Hugging Face weights) and adjacent-pair (GGUF files converted for llama.cpp usually permute Q/K so adjacent-pair applies). Select by flag and verify against the model you actually load in Module 7. A mismatch here gives plausible-looking garbage.
- [ ] **RoPE tables.** Precompute cos/sin in double on the host, store FP32. Avoid on-the-fly `__sinf` at large positions.
- [ ] **Workspace.** Decode partials use `ScopedPoolBuffer` with stream-ordered free.
- [ ] **Scale.** `scale` is a parameter (default `1/sqrt(D)`), applied in FP32.

## 4. Phases

### Phase 0: Pre-flight gate (about 2 days)
- [ ] **[Must]** Commit the pending 17-file tree in small commits (argmax; allocator and stream safety; `K == 0`; golden generators and tests; harness), building with `-DENGINE_WERROR=ON` and running `ctest` after each.
- [ ] **[Must]** Shrink golden data (the 747 MiB is mostly oversized embedding tables). Cross-check the NumPy and Torch backends of `gen_reference.py` agree to about 1e-12.
- [ ] **[Must]** Regenerate `results.json` from clean `HEAD` with warm-up and clock logging; check sampled clocks and IQR before trusting it.
- [ ] **[Should]** Generate doc test counts from the runner (`--list`), not by hand.
- *Done when:* ctest passes on Machine B, CI is green on all five checks, and `results.json` matches `HEAD`.

### Phase 1: Design (1 day)
- [ ] **[Must]** Write ADR-0012 (layouts, API, precision, contracts) and a stub for ADR-0013 (decode split).
- [ ] **[Must]** Notebook entry with predictions: naive versus flash memory and time at S=512, 2048; decode time at L=2048 and 8192.
- *Done when:* the ADR is reviewed by you with a fresh head the next morning.

### Phase 2: References and test infrastructure (2 days)
- [ ] **[Must]** CPU oracle `attention_cpu`: double accumulation, standard two-pass softmax, supports mask, GQA, and offset.
- [ ] **[Must]** A second scalar oracle that mirrors the kernel's online-softmax recurrence. If it matches Tier 1 but the GPU doesn't, the bug is in the CUDA mapping, not the math.
- [ ] **[Must]** `rope_cpu` oracle for both conventions.
- [ ] **[Must]** Tier 1 generators in `tools/gen_reference.py` for attention and RoPE (float64, computed from the FP16-rounded inputs so only kernel error is measured). Keep cases small.
- [ ] **[Must]** Derived tolerance helper from FP16 epsilon and `s_kv`, the same method as the `gemv_fp16` tests. Never hand-tune.
- *Done when:* oracles agree with Tier 1 across the test matrix in Section 5.

### Phase 3: Naive baseline (2 days)
- [ ] **[Must]** Naive FP16 attention on GPU: scores kernel (FP32 scores buffer, mask written as `-inf`), existing `softmax_rows`, then a P·V kernel. Slow on purpose.
- [ ] **[Must]** Tests against Tier 1 and the CPU oracles.
- [ ] **[Must]** Benchmark naive at S in {128, 512, 1024, 2048} to get the baseline and the scores-buffer footprint.
- *Done when:* naive passes the full matrix. It is now your GPU-side oracle for the flash kernels.

### Phase 4: RoPE (1.5 days)
- [ ] **[Must]** RoPE kernel (in place on Q and K), both conventions, table-driven, position start as a parameter.
- [ ] **[Must]** Golden tests including position 0, mid-range, 2047, and one very large position (for example 100,000) to expose precision errors.
- [ ] **[Should]** Benchmark; at decode sizes it is latency-bound, so report latency, not bandwidth.
- *Done when:* both conventions match Tier 1.

### Phase 5: FlashAttention-2 prefill (about 5 days)
- [ ] **[Must]** v1 on CUDA cores: grid over `(ceil(s_q / Br), n_heads)`, query tile resident, loop over KV tiles inside, rows split across warps (the FA2 change that avoids cross-warp reductions). Start with Br = Bc = 64 at D=64; use Bc = 32 or opt-in dynamic shared memory at D=128 (3 tiles of 16 KiB hit the 48 KiB static limit).
- [ ] **[Must]** Online softmax with FP32 running max and sum, FP32 output accumulator, a single final rescale.
- [ ] **[Must]** Causal tile skipping: skip tiles fully above the diagonal, mask only diagonal tiles. Test with `q_pos_offset` values that are not tile multiples.
- [ ] **[Must]** GQA mapping `kv_head = h / group`.
- [ ] **[Should]** Optional logsumexp output for debugging against the oracle.
- [ ] **[Should]** GQA group sharing: process the group's heads in one block so K/V tiles are loaded once (8x fewer loads for TinyLlama). Measure with and without.
- [ ] **[Should]** Check `-Xptxas -v` for register spills; record occupancy.
- [ ] **[Stretch]** v2 with tensor cores (WMMA or `mma.sync`) for QK^T and PV, with its own prediction and before/after entry.
- Notebook rule: prediction first, one entry per optimization step, negative results go in `negative-results.md`.
- *Done when:* flash matches naive and Tier 1 across the matrix, and uses O(S) memory (no score matrix).

### Phase 6: Decode attention and FlashDecoding (about 4 days)
- [ ] **[Must]** Decode kernel for `s_q = 1`: one pass over the cached K/V with online softmax. It is bandwidth-bound, so judge it against the clock-adjusted memory peak.
- [ ] **[Must]** Split-KV: the grid `(n_kv_heads, num_splits)` has far fewer blocks than your 56 SMs at TinyLlama (4 KV heads), so split the sequence across blocks, write per-split `(m, l, O)` partials to pool workspace, then combine with a log-sum-exp reduction kernel. Starting heuristic: `num_splits ~ 4 * SMs / n_kv_heads`, clamped by `kv_len / min_chunk`. Sweep it.
- [ ] **[Must]** Each block handles all query heads of its KV group, so K/V are read once per group.
- [ ] **[Must]** Tests: `splits=1` equals `splits=N`; odd `kv_len`; `kv_len < splits`; `kv_len = 1`; multi-stream use of the workspace.
- [ ] **[Should]** Write ADR-0013 with the measured split sweep.
- *Done when:* decode matches the oracle and the split sweep shows where splitting stops helping.

### Phase 7: Integration test (1.5 days)
- [ ] **[Must]** One attention sub-block end to end: RMSNorm, QKV projection (`gemv` and register-tiled GEMM), RoPE, attention, output projection, residual add. Compare with a float64 NumPy golden at reduced dims plus one real TinyLlama-shaped case.
- *Done when:* it passes at prefill and decode shapes. This previews the layer-by-layer comparison tool in Module 8.

### Phase 8: Benchmarks and analysis (2 days)
Protocol (same as the rest of the repo): warm-up, per-row minimum warm-up, sampled live clocks, median and IQR, cold cache (flush L2 or rotate at least 3x L2 of buffers), report % of rated peak and % of clock-adjusted peak.
- [ ] **[Must]** Prefill: naive versus flash, S in {128, 512, 1024, 2048, 4096}, time, causal-convention TFLOP/s (state it), and memory footprint versus S. Find the crossover.
- [ ] **[Must]** Decode: L in {512, 2048, 8192, 16384}, split sweep, GB/s, rotating KV buffers. Below the size where the kernel is latency-bound, report latency instead of bandwidth.
- [ ] **[Must]** GQA ablation: measured traffic and time with and without group sharing.
- [ ] **[Should]** Nsight Compute on flash prefill and decode: DRAM bytes versus your prediction, bank conflicts, stall reasons. Add attention points to the roofline plot.
- [ ] **[Stretch]** Run PyTorch SDPA from a script in `tools/` as an external yardstick, never linked into the engine.

### Phase 9: Close-out (1.5 days)
- [ ] **[Must]** ADRs finalized, notebook complete (no past numbers edited), README table regenerated from `results.json`, `brain/STATUS.md` and test counts generated from the runner.
- [ ] **[Must]** CI green on all five checks (including MSVC: no GCC-only builtins).
- [ ] **[Must]** Commit at the end of each phase.

## 5. Test matrix

| Axis | Values |
|---|---|
| Sequence length `s_q`, `s_kv` | 1, 2, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 1000, 2048 |
| Head dim | 64, 128 |
| (`n_heads`, `n_kv_heads`) | (1,1), (8,2), (32,4), (32,32) |
| Causal | on, off |
| `q_pos_offset` | 0, 5, 1000 |
| Logit stress | large magnitude (scaled to about ±30 after scaling), all-equal scores, one dominant key, single valid key (first causal row) |
| Contracts | `s_kv = 0` gives zeros, `s_q = 0` is a no-op, bad head ratio throws, unsupported head dim throws, null pointers |
| RoPE | positions 0, mid, 2047, 100,000; both conventions |

Compare every flash and decode result against Tier 1 (float64), against naive, and (decode) across split counts.

## 6. Files to add or change

- `kernels/attention_naive.cu`, `kernels/attention_prefill.cu`, `kernels/attention_decode.cu`, `kernels/rope.cu`
- `include/engine/kernels.hpp` (declarations and contracts, update stale docstrings while there)
- `src/cpu_ref/attention_cpu.cpp`, `src/cpu_ref/rope_cpu.cpp`
- `tests/test_kernels.cu`, `tests/test_cpu_ref.cpp`
- `bench/bench_kernels.cu`, `tools/gen_reference.py`, `tools/roofline_plot.py`
- `docs/adr/0012-*.md`, `docs/adr/0013-*.md`, `docs/lab-notebook.md`, `docs/negative-results.md`, `plans/module4_plan.md`

## 7. Schedule (assumes steady daily work; estimates are rough)

| Dates | Phase |
|---|---|
| Nov 11-13 | 0: pre-flight gate |
| Nov 14 | 1: design |
| Nov 15-16 | 2: references and tests |
| Nov 17-18 | 3: naive baseline |
| Nov 19-20 | 4: RoPE |
| Nov 21-26 | 5: flash prefill |
| Nov 27-30 | 6: decode (starts here; finishes early December) |
| Dec 1-3 | 6 finish, 7: integration, 8: benchmarks, 9: close-out |

The Must-only path fits close to November 30 if nothing slips. The tensor-core path, group sharing, and Nsight analysis are what push it into the first week of December. Module 5 does not need to start before the gate in Section 8 passes.

## 8. Risks and cut order

| Risk | Mitigation |
|---|---|
| Online-softmax numerical bugs | Scalar recurrence oracle plus adversarial logit tests |
| RoPE convention mismatch with GGUF | Both modes, golden tests, verify on the real model in Module 7 |
| Shared memory limit at D=128 | Smaller `Bc` or dynamic shared memory opt-in |
| Register spills | Check `-Xptxas -v` early, adjust tile sizes |
| Decode underuses the GPU (32 or fewer blocks on 56 SMs) | Split-KV, with a split sweep |
| Causal tile-boundary bugs | Offsets that are not tile multiples, sequence lengths around 32, 64, 128 |
| Time | Cut in this order: PyTorch yardstick, tensor-core v2, group-sharing optimization, Nsight depth. Never cut: naive baseline, correctness matrix, causal and GQA, split-KV decode, benchmark protocol |

## 9. Definition of done

- Full test matrix passes on the GPU machine, plus Tier 1 golden coverage for attention and RoPE, with counts generated by the runner.
- Flash prefill uses O(S) memory and matches naive and Tier 1; the crossover with naive is measured.
- Decode attention has a measured split sweep and reports bandwidth against the clock-adjusted peak with cold KV data.
- The GQA saving is measured, not estimated.
- Initial performance targets, to be revised in the prediction entry before measuring: flash v1 at least 25% of FP32 peak at S=2048; decode at least 80% of clock-adjusted peak at large `kv_len`.
- ADR-0012 and ADR-0013 written, notebook entries have predictions before measurements, CI green, `results.json` regenerated from the committed `HEAD`.
