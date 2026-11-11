# Module 4 Phase 10: Full Llama Transformer Block Integration

**Written:** 2026-11-11 | **Base:** commit `92b7c86` (Module 4 Phases 0–9 complete) | **GPU:** RTX 4070 SUPER (sm_89, 56 SMs, 504 GB/s peak DRAM BW) | **Precision:** FP32 weights & norms, FP16 attention I/O, FP32 accumulation

Priority tags: **[Must]** core to the result, **[Should]** strengthens it, **[Stretch]** cut first.

---

## 1. Goal

Phase 10 closes a critical gap before Module 5 (Quantization): every kernel needed for a Llama transformer layer exists and passes standalone tests, but **they have never been composed into a complete block**. Only the Attention Sub-Block pipeline is currently integrated (in `tests/test_kernels.cu` L2116–L2188). The FFN Sub-Block (`RMSNorm → Gate/Up Projections → SwiGLU → Down Projection → Residual`) and the combined Attention+FFN Transformer Block remain unwired.

Phase 10 should answer one question: **Does the full transformer block produce bit-exact (within FP16 tolerance) output when all kernels are composed end-to-end, including multi-step autoregressive KV cache reuse across both sub-blocks?**

**In scope:** FFN sub-block integration, full transformer block (Attention + FFN), multi-step prefill→decode test with shared KV cache, float64 golden generation via PyTorch, CPU oracle path, GPU path, tolerance derivation, optional 2-layer stacking test.

**Out of scope:** New CUDA kernels (all exist), model loading, quantization, multi-layer KV-cache management class, benchmarking/profiling (this is a correctness-only phase).

## 2. Architecture: Llama Transformer Block

A single LlamaDecoderLayer at TinyLlama-1.1B dimensions (`d_model=2048`, `n_heads=32`, `n_kv_heads=4`, `head_dim=64`, `d_ffn=5632`):

```
Input x [S, 2048]
│
├─── Attention Sub-Block (EXISTING — Phase 7) ─────────────────────────────┐
│   1. attn_norm = RMSNorm(x, w_attn_norm, eps)           [S, 2048]      │
│   2. Q = attn_norm × W_q                                 [S, 2048]      │
│      K = attn_norm × W_k                                 [S, 256]       │
│      V = attn_norm × W_v                                 [S, 256]       │
│   3. Q_f16, K_f16, V_f16 = cast_f32_to_f16(Q, K, V)                    │
│   4. RoPE(Q_f16, K_f16, cos, sin, pos_offset)                           │
│   5. kv_cache_write(K_f16, V_f16 → k_cache, v_cache)                    │
│   6. if S > 1: attn_out = attention_prefill(Q, K_cache, V_cache, ...)   │
│      else:     attn_out = attention_decode(Q, K_cache, V_cache, ...)    │
│   7. attn_f32 = cast_f16_to_f32(attn_out)                               │
│   8. o_proj = attn_f32 × W_o                              [S, 2048]     │
│   9. h = x + o_proj                       (residual add)  [S, 2048]     │
│                                                                          │
├─── FFN Sub-Block (NEW — Phase 10) ───────────────────────────────────────┤
│  10. ffn_norm = RMSNorm(h, w_ffn_norm, eps)               [S, 2048]     │
│  11. gate = ffn_norm × W_gate                              [S, 5632]     │
│      up   = ffn_norm × W_up                                [S, 5632]     │
│  12. act = SwiGLU(gate, up)                                [S, 5632]     │
│  13. down = act × W_down                                   [S, 2048]     │
│  14. out = h + down                       (residual add)   [S, 2048]    │
│                                                                          │
Output out [S, 2048] ◄────────────────────────────────────────────────────┘
```

### Key dimensions

| Model | `d_model` | `n_heads` | `n_kv_heads` | `head_dim` | `d_ffn` (intermediate) | GQA ratio |
|---|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | 2048 | 32 | 4 | 64 | 5632 | 8:1 |
| Llama-2-7B | 4096 | 32 | 32 | 128 | 11008 | 1:1 (MHA) |

Phase 10 tests at **reduced dimensions** for golden file size control, plus one test at **exact TinyLlama dimensions** (same pattern as the existing attention sub-block tests).

## 3. Decisions and constraints

These are engineering constraints, not new ADRs. All kernel APIs and layouts are already decided (ADR-0003, ADR-0012, ADR-0013).

- **Weight precision:** FP32 for all projection weights (`W_q`, `W_k`, `W_v`, `W_o`, `W_gate`, `W_up`, `W_down`) and RMSNorm weights. This matches the real inference path where GGUF FP16/quantized weights are dequantized to FP32 before GEMM/GEMV. Module 5 will swap in fused dequant+GEMV/GEMM.
- **Activation precision:** FP32 throughout the residual stream (`x`, `h`, `out`), with FP16 only inside the attention sub-block (Q/K/V after cast, attention output before cast back). SwiGLU uses the FP32 variant (`engine::cuda::swiglu`, not `swiglu_fp16`), matching the residual-stream precision.
- **Projection dispatch:** `matmul_register_tiled` for prefill (`s_q > 1`), `gemv` for decode (`s_q == 1`). Same pattern as existing `run_attention_subblock_gpu`.
- **No `rmsnorm_linear` fusion in Phase 10.** Use separate `rmsnorm` + `matmul`/`gemv` for clarity and to match the oracle exactly. Fused dispatch is a performance concern for Module 8 benchmarking.
- **Tolerance derivation:** The transformer block chains two sub-blocks. Each introduces FP16 truncation at Q/K/V casts, FP16→FP32 output cast, and FP32 accumulation noise. The FFN sub-block is all-FP32 (no FP16 truncation boundaries), so its error contribution is small. The block tolerance should be derived, not hand-tuned — documented in a comment like the existing `subblock_rtol()` / `subblock_atol()`.
- **KV cache ownership:** Raw `DeviceBuffer<half>` / `std::vector<half>` per test, same as the existing attention sub-block tests. Module 6 will introduce a proper per-layer KV-cache manager.

## 4. Sub-phases

### Sub-phase 10a: FFN Sub-Block Integration (~0.5 day)

**No new kernels — only composition of existing tested kernels.**

- [ ] **[Must]** `run_ffn_subblock_gpu()` function in `tests/test_kernels.cu`: chains `rmsnorm` → 2× `matmul_register_tiled`/`gemv` (gate, up) → `swiglu` → `matmul_register_tiled`/`gemv` (down) → `vector_add` (residual). Parameters: `(d_h, d_ffn_norm_w, d_w_gate, d_w_up, d_w_down, d_out, s_q, d_model, d_ffn, eps)`.
- [ ] **[Must]** `run_ffn_subblock_cpu()` matching function using CPU oracles: `cpu::rmsnorm` → `cpu::matmul` (gate, up) → `cpu::swiglu` → `cpu::matmul` (down) → `cpu::vector_add`.
- [ ] **[Must]** `run_ffn_subblock()` in `tools/gen_reference.py`: float64 reference pipeline using `backend.rmsnorm`, `backend.matmul`, `backend.swiglu`, `backend.vector_add`.
- [ ] **[Must]** Derive FFN sub-block tolerance. Since the entire path is FP32 (no FP16 truncation), tolerance should be tighter than the attention sub-block — approximately `4 * eps32 * sqrt(d_model)` relative. Document the derivation in a `ffn_subblock_rtol()` / `ffn_subblock_atol()` helper.

### Sub-phase 10b: FFN Sub-Block Golden Tests (~0.5 day)

- [ ] **[Must]** Golden cases in `gen_reference.py`:

  | Name | `s_q` | `d_model` | `d_ffn` | Purpose |
  |---|---:|---:|---:|---|
  | `ffn_subblock__prefill_s16_d512_ffn1408` | 16 | 512 | 1408 | Reduced-dim golden (small file) |
  | `ffn_subblock__decode_s1_d512_ffn1408` | 1 | 512 | 1408 | Decode dispatch path |

- [ ] **[Must]** `TEST(kernels, ffn_subblock_matches_reference)`: GPU vs Tier-1 golden across the above cases.
- [ ] **[Must]** `TEST(cpu_ref, ffn_subblock_matches_reference)`: CPU oracle vs Tier-1 golden.

### Sub-phase 10c: Full Transformer Block Composition (~0.5 day)

- [ ] **[Must]** `run_transformer_block_gpu()` function: calls `run_attention_subblock_gpu()` followed by `run_ffn_subblock_gpu()`, passing the attention output `h` as the FFN input.
- [ ] **[Must]** `run_transformer_block_cpu()` matching function.
- [ ] **[Must]** `run_transformer_block()` in `gen_reference.py`: float64 reference calling `run_attention_subblock()` then `run_ffn_subblock()`.
- [ ] **[Must]** Derive full-block tolerance. Two sub-blocks chained: attention sub-block tolerance + FFN sub-block tolerance (additive, since FFN operates on the attention output). Should be approximately `block_rtol = subblock_rtol + ffn_rtol` and `block_atol = subblock_atol + ffn_atol`. Document the derivation.

### Sub-phase 10d: Transformer Block Golden Tests (~0.5 day)

- [ ] **[Must]** Golden cases:

  | Name | `s_q` | `past_len` | `n_heads` | `n_kv_heads` | `head_dim` | `d_ffn` | Purpose |
  |---|---:|---:|---:|---:|---:|---:|---|
  | `transformer_block__prefill_s16_h8_kv2_d64_ffn1408` | 16 | 0 | 8 | 2 | 64 | 1408 | Reduced-dim prefill |
  | `transformer_block__decode_s1_kv17_h8_kv2_d64_ffn1408` | 1 | 16 | 8 | 2 | 64 | 1408 | Decode with past KV from prefill case |

- [ ] **[Must]** `TEST(kernels, transformer_block_prefill_and_decode_golden)`: GPU vs Tier-1 across both cases.

### Sub-phase 10e: TinyLlama-Shape Multi-Step Test (~0.5 day)

This is the headline integration test — same pattern as `TEST(kernels, attention_subblock_tinyllama_shape)`.

- [ ] **[Must]** `TEST(kernels, transformer_block_tinyllama_shape)`:
  1. **Step 0 — Prefill:** `s_q=16`, `past_len=0`. Run full transformer block at exact TinyLlama dimensions (`d_model=2048`, `n_heads=32`, `n_kv_heads=4`, `head_dim=64`, `d_ffn=5632`). Compare GPU vs CPU oracle.
  2. **Step 1 — Decode:** `s_q=1`, `past_len=16` (`kv_len=17`). Reuses the KV cache populated during prefill. Compare GPU vs CPU oracle.
  3. **Step 2 — Decode:** `s_q=1`, `past_len=17` (`kv_len=18`). Verifies KV cache growth across the full block. Compare GPU vs CPU oracle.

  This test does NOT use float64 Tier-1 goldens (weights would be too large). Instead it uses GPU-vs-CPU agreement at the derived block tolerance, exactly like the existing `attention_subblock_tinyllama_shape` test which uses `pseudo_normal_vec` for deterministic weights.

- [ ] **[Should]** `TEST(kernels, transformer_block_two_layers)`: Run 2 layers sequentially, feeding output of layer 0 into layer 1, each with independent weights and a shared KV cache per layer. Verify GPU-vs-CPU agreement. This catches residual-stream precision accumulation across layers and validates that the KV cache pattern works for N>1 layers.

### Sub-phase 10f: Close-out (~0.25 day)

- [ ] **[Must]** `ctest --test-dir build` passes (all suites, including new tests).
- [ ] **[Must]** `ctest --test-dir build-cpu` passes (CPU-only GCC build).
- [ ] **[Must]** `clang-format --dry-run --Werror` passes on all tracked files.
- [ ] **[Must]** Update `brain/STATUS.md` with new test count and transformer block completion note.
- [ ] **[Must]** Commit with descriptive message.
- [ ] **[Should]** Update `README.md` test count if it has changed.

## 5. Test matrix

| Test | What it verifies | Golden source | Tolerance |
|---|---|---|---|
| `ffn_subblock_matches_reference` | FFN sub-block GPU vs Tier-1 (prefill + decode shapes) | Float64 via `gen_reference.py` | `ffn_subblock_rtol/atol` (FP32-only path, tight) |
| `cpu_ref ffn_subblock_matches_reference` | FFN sub-block CPU oracle vs Tier-1 | Float64 via `gen_reference.py` | Same |
| `transformer_block_prefill_and_decode_golden` | Full block GPU vs Tier-1 (prefill + decode, reduced dims) | Float64 via `gen_reference.py` | `block_rtol/atol` (derived, documented) |
| `transformer_block_tinyllama_shape` | Full block GPU vs CPU at exact TinyLlama dims, multi-step | GPU-vs-CPU agreement | `block_rtol/atol` |
| `transformer_block_two_layers` `[Should]` | 2-layer residual precision + KV cache isolation | GPU-vs-CPU agreement | `block_rtol/atol` (may need `2×` for 2 layers) |

## 6. Files to add or change

| File | Change |
|---|---|
| `tools/gen_reference.py` | Add `run_ffn_subblock()`, `run_transformer_block()`, `FFN_SUBBLOCK_CASES`, `TRANSFORMER_BLOCK_CASES`, and golden generation loops |
| `tests/test_kernels.cu` | Add `run_ffn_subblock_gpu()`, `run_transformer_block_gpu()`, and 3–4 new `TEST()` functions |
| `tests/test_cpu_ref.cpp` | Add `run_ffn_subblock_cpu()`, `run_transformer_block_cpu()`, and 1 new `TEST()` function |
| `brain/STATUS.md` | Update test count and add Phase 10 completion note |
| `README.md` | Update test count if changed |

No new headers, no new kernel files, no new `.cpp` under `src/` — this is purely a test/verification phase composing existing APIs.

## 7. Golden file size estimate

| Golden | Elements | Size (float64) |
|---|---:|---:|
| `ffn_subblock__prefill_s16_d512_ffn1408` | x(`8K`) + w_norm(`512`) + W_gate(`720K`) + W_up(`720K`) + W_down(`720K`) + expected(`8K`) ≈ `2.17M` | ~16.5 MiB |
| `ffn_subblock__decode_s1_d512_ffn1408` | Same weights + x(`512`) + expected(`512`) ≈ `2.16M` | ~16.5 MiB |
| `transformer_block__prefill` | Attn weights (`3 × 512×128` + `512×512`) + FFN weights (`3 × 512×1408`) + norms + RoPE + expected ≈ `2.4M` | ~18.5 MiB |
| `transformer_block__decode` | Same weights + past KV + expected ≈ `2.4M` | ~18.5 MiB |
| **Phase 10 total addition** | | **~70 MiB** |
| **Projected corpus total** | Current `66.83 MiB` + `~70 MiB` | **~137 MiB** |

> [!NOTE]
> The FFN weight matrices are the dominant cost (`512 × 1408 × 3 = ~2.15M floats` per case). Using reduced `d_model=512, d_ffn=1408` (¼ of TinyLlama) keeps each golden under ~18 MiB. The TinyLlama-shape test avoids goldens entirely (GPU-vs-CPU at `2048 × 5632` weights would be ~350 MiB per case).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| FFN sub-block tolerance too tight (FP32 matmul accumulation noise at `d_model=2048`) | Derive from `eps32 * sqrt(K)` for each projection; if GPU-vs-CPU disagrees, loosen with documented arithmetic — never hand-tune |
| TinyLlama FFN weight allocation exceeds GPU memory at `d_model=2048, d_ffn=5632` | Each projection is at most `2048 × 5632 × 4 = 44 MiB`; total block weights < 250 MiB; well within 12 GiB VRAM |
| Golden file corpus doubles | Acceptable for integration verification; Module 5+ will NOT add more golden FFN files since quantization tests will compare against the FP32 block output directly |
| SwiGLU FP32 vs FP16 mismatch | Phase 10 uses `engine::cuda::swiglu` (FP32 variant) for the FFN path, matching the all-FP32 residual stream. No FP16 SwiGLU in the block pipeline |

## 9. Schedule

| Effort | Sub-phase |
|---|---|
| ~0.5 day | 10a: FFN sub-block GPU + CPU functions |
| ~0.5 day | 10b: FFN sub-block golden tests |
| ~0.5 day | 10c: Transformer block composition |
| ~0.5 day | 10d: Transformer block golden tests |
| ~0.5 day | 10e: TinyLlama-shape multi-step + [Should] 2-layer test |
| ~0.25 day | 10f: Close-out (ctest, format, commit) |
| **Total** | **~2.75 days** |

## 10. Definition of done

- FFN sub-block passes at both prefill and decode shapes against Tier-1 float64 goldens, with derived (not hand-tuned) tolerance documented in code comments.
- Full transformer block (Attention + FFN) passes at reduced dimensions against Tier-1 float64 goldens.
- Multi-step test at exact TinyLlama-1.1B dimensions (`d_model=2048, d_ffn=5632, n_heads=32, n_kv_heads=4, head_dim=64`) runs prefill → decode → decode with shared KV cache and passes GPU-vs-CPU agreement.
- All existing `141` tests still pass plus the new Phase 10 tests.
- `clang-format` and CPU-only CTest suite remain clean.
- `brain/STATUS.md` updated with new test count.
- Committed at a clean HEAD ready for Module 5.

