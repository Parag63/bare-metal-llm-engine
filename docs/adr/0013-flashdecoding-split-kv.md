# ADR 0013 — FlashDecoding Split-KV Heuristic and GQA Head Sharing (Stub)

- **Status:** Proposed (Stub — to be finalized after Phase 6 Split-KV & GQA sweep)
- **Date:** 2026-11-11
- **Module:** Module 4 (FlashDecoding / Decode Attention)
- **Builds on:** [ADR 0011](0011-flashattention2-fp16-precision.md), [ADR 0012](0012-attention-layouts-and-api.md)

---

## Context

In single-token autoregressive decode (`s_q = 1`), standard FlashAttention-2 parallelizes only across `(n_heads, ceil(1 / Br))` = `n_heads` thread blocks (e.g., 32 blocks for TinyLlama-1.1B and Llama-2-7B, or only 4 blocks if scheduled per `n_kv_heads`). On an RTX 4070 SUPER (`sm_89`, **56 SMs**), launching `<= 32` blocks leaves >= 43% of SMs completely idle and lacks active warps per SM to hide GDDR6X latency, capping achieved KV-cache streaming bandwidth well below 504 GB/s.

**FlashDecoding** partitions `kv_len` into `split_kv` chunks along the sequence dimension so `n_heads * split_kv` (or `n_kv_heads * split_kv` with GQA group sharing) blocks keep all 56 SMs busy, then reduces partial `(m, l, O)` states via a second lightweight LSE reduction kernel using stream-ordered `ScopedPoolBuffer` workspace.

---

## Planned Experiments (Phase 6 & Phase 8)

This ADR will record the final empirical threshold rules after running Phase 6 sweeps on RTX 4070 SUPER (`sm_89`):

1. **Split-KV (`split_kv`) selection heuristic** as a function of `kv_len` (spanning `L in {1, 31, 64, 512, 2048, 8192, 16384}`), `n_heads`, `n_kv_heads`, and `num_sms = 56`:
   - Overhead crossover where split + reduction beats single-pass (`split_kv == 1`).
   - Optimal block count / minimum tokens per split (`min_chunk_len`) vs workspace memory traffic.
2. **GQA KV-load sharing across query heads within a group** (`group = n_heads / n_kv_heads`, e.g., TinyLlama `group = 8`, `32 / 4`):
   - Comparing per-Q-head grid `(n_heads * splits)` vs per-KV-head GQA-group grid `(n_kv_heads * splits)` for both prefill (`attention_prefill`) and decode (`attention_decode`).
3. **Final Workspace Formula & Default Policy:** Exact analytical rule encoded in `attention_decode(..., split_kv = 0, stream)`.

*(Results and final decision will populate this document upon Phase 6 completion.)*
