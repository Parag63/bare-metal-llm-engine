# ADR 0013 — FlashDecoding Split-KV Heuristic and GQA Head Sharing

- **Status:** Accepted
- **Date:** 2026-11-11
- **Module:** Module 4 (FlashDecoding / Decode Attention)
- **Builds on:** [ADR 0011](0011-flashattention2-fp16-precision.md), [ADR 0012](0012-attention-layouts-and-api.md)

---

## Context

In single-token autoregressive decode (`s_q = 1`), standard attention parallelizes only across `n_heads` (or `n_kv_heads` when sharing KV loads within a GQA group). On an RTX 4070 SUPER (`sm_89`, **56 SMs**):

- TinyLlama-1.1B (`H_q=32, H_kv=4, D=64`) launches only **4 blocks** if scheduled per `n_kv_heads`, leaving **92.9% of the GPU idle** (`~21.5 GB/s`, 4.3% of 504 GB/s peak).
- Even Llama-2-7B (`H_q=32, H_kv=32, D=128`) launches only **32 blocks** (`57%` of SMs), leaving 24 SMs idle at `splits=1`.

**FlashDecoding** splits `kv_len` across `actual_splits` blocks along `grid.y` and combines partial `(m, l, O)` states via a lightweight LSE reduction pass backed by a stream-ordered `ScopedPoolBuffer`.

---

## Architectural Design & Investigation

### 1. Warp-Per-Head SMEM-KV Broadcast (`G in {2, 4, 8}`) vs Thread-Local `kGChunk` Loop

Two GQA group-sharing strategies were evaluated in [kernels/attention_decode.cu](../../kernels/attention_decode.cu):

- **Rejected primary GQA path (`flash_decode_split_kernel` with `kGChunk=8`):** Each warp streams `K/V` directly into thread registers and loops serially over `8` query heads per key (`m_loc[8]`, `l_loc[8]`, `o_loc[8][8]`), followed by 8 sequential cross-warp SMEM reductions. While it avoids KV reload traffic, it consumes **134–138 registers/thread** (`1` active block/SM = 8 warps/SM = **16.7% occupancy**), saturating TinyLlama `L=16384` at `0.096 ms` (`174.4 GB/s`) and losing at `L=2048` (`0.037 ms` vs `0.023 ms` unshared). Retained only as a fallback for non-power-of-two `group_size` or `group_size > 8` and for `group_size == 1` MHA (where `67–70` regs reach 50% occupancy and `463.6 GB/s`).
- **Accepted GQA kernel (`flash_decode_gqa_warp_kernel` for `G in {2, 4, 8}`):** Launches `G` warps per block (`1` warp owns `1` query head held in FP32 registers `q_f[8]`). All warps cooperatively load a `kTileK` tile of `K` and `V` (`32` keys for `G=8, D=64`: 8 KiB SMEM total) once from DRAM using `LDG.128` (`uint4`), then each warp computes its own head independently with zero cross-warp SMEM reduction:
  - Registers drop to **53–54** (`G=8, D=64`) / **48** (`G=2,4, D=64`), lifting SM occupancy **4×** (`4` active blocks/SM = 32 warps/SM = **66.7%**), **0 B stack / 0 B spills**.
  - TinyLlama `L=16384` improves **1.92×** (`0.096 ms -> 0.050 ms`, `174.4 -> 335.4 GB/s`) and beats unshared `no GQA share` across `L=2048..32768` by **1.63–2.10×**.

### 2. Split-KV Sweep (`sm_89`, 56 SMs, 2790 MHz SM / 10251 MHz Mem)

Sweeping TinyLlama (`H_q=32, H_kv=4, D=64`) with cold 256 MiB rotating KV slabs:

| `kv_len` | `splits=1` | `splits=4` | `splits=8` | `splits=14` | `splits=16` | `splits=28` | `splits=32` | `splits=42` | `splits=56` | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `128` | 0.0100 ms | **0.0081 ms** | 0.0082 ms | 0.0092 ms | 0.0092 ms | 0.012 ms | 0.012 ms | 0.013 ms | 0.017 ms | `1` (avoids reduce) |
| `512` | 0.029 ms | 0.013 ms | **0.011 ms** | 0.012 ms | 0.012 ms | 0.013 ms | 0.013 ms | 0.015 ms | 0.017 ms | `4–8` |
| `2048` | 0.101 ms | 0.031 ms | 0.020 ms | **0.016 ms** | **0.016 ms** | 0.018 ms | 0.018 ms | 0.019 ms | 0.021 ms | `14` (1 full wave) |
| `8192` | 0.392 ms | 0.104 ms | 0.057 ms | 0.039 ms | 0.039 ms | **0.032 ms** | **0.032 ms** | 0.033 ms | 0.035 ms | `28` (265.3 GB/s) |
| `16384` | 0.779 ms | 0.201 ms | 0.105 ms | 0.067 ms | 0.069 ms | **0.050 ms** | 0.051 ms | 0.051 ms | 0.054 ms | `28` (334.5 GB/s) |
| `32768` | 1.56 ms | 0.398 ms | 0.204 ms | 0.125 ms | 0.128 ms | **0.088 ms** | 0.090 ms | **0.088 ms** | 0.090 ms | `28` (381.5 GB/s) |

Notice the SM-wave alignment effect for `y_blocks = 4` on 56 SMs (`per_wave = 56 / 4 = 14`): `splits=16` (`64` blocks = 56 + 8 tail) is consistently slower than `splits=14` (`56` blocks = 1 full wave), and `splits=28` (`112` blocks = 2 full waves) beats `splits=32` (`128` blocks = 112 + 16 tail) as well as over-split `56`.

---

## Decision (`split_kv == 0` Heuristic & Workspace Formula)

When callers pass `split_kv == 0` (`auto`), `choose_auto_splits(kv_len, y_blocks)` computes:

1. **Short-sequence cutoff:** If `kv_len < 256`, return `1` (skip scratch allocation and reduce launch).
2. **Occupancy target:** `target_blocks = 2 * num_sms` (`112` blocks on 56 SMs), so `wanted = ceil(target_blocks / y_blocks)`.
3. **Amortization floor (`min_keys_per_split`):** `128` keys/split when `y_blocks * 2 < num_sms` (GQA-starved), or `512` keys/split when `y_blocks * 2 >= num_sms` (`H_kv=32` already covers 57% of SMs at `splits=1`, preventing `Llama2-7B L=512` from regressing via over-splitting: `0.023 ms` at `splits=1` vs `0.027 ms` with extra splits). Clamp `splits <= min(wanted, kv_len / min_keys_per_split, 64)`.
4. **SM-wave alignment:** If `num_sms % y_blocks == 0` and `per_wave = num_sms / y_blocks > 1` and `splits > per_wave`, snap `splits = (splits / per_wave) * per_wave`.
5. **Stream-ordered workspace:** When `actual_splits > 1`, allocate `ScopedPoolBuffer` of `n_heads * actual_splits * (head_dim + 2) * sizeof(float)` bytes (~228 KiB for TinyLlama at `splits=28`, ~130 KiB for Llama-2-7B at `splits=4`), and run `flash_decode_reduce_kernel` across `n_heads` blocks. When `actual_splits == 1`, emit FP16 directly with zero workspace and one kernel launch.

---

## Consequences

- **TinyLlama-1.1B (`32/4, D=64`):** Speeds up decode from `0.781 ms -> 0.050 ms` (**15.6×**) at `L=16384` (`335.4 GB/s`, 66.5% peak BW) and `1.56 ms -> 0.088 ms` (**17.7×**) at `L=32768` (`381.1 GB/s`, 75.6% peak BW), outperforming unshared `no GQA share` by **2.10×** at `L=32768`.
- **Llama-2-7B (`32/32, D=128`):** Hits **`431.4 GB/s` (85.6%)** at `L=2048` and **`463.6 GB/s` (91.97% peak BW)** at `L=8192`, while avoiding unnecessary splits at `L=512` (`0.023 ms`, `362.0 GB/s`).
- **Determinism & Testing:** Explicit `split_kv > 0` lets callers and tests reproduce exact split reductions (tested across `splits in {1, 2, 3, 4, 8, 16, 32}`, including `kv_len < split_kv`, odd `kv_len`, unaligned offsets, and multi-stream pool safety).
