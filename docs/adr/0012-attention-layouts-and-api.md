# ADR 0012 — Attention Tensor Layouts, Kernel API, and Numerical Contracts

- **Status:** Accepted
- **Date:** 2026-11-11
- **Module:** Module 4 (Attention, RoPE, FlashAttention-2, FlashDecoding)
- **Builds on:** [ADR 0003](0003-raw-pointer-kernel-api.md), [ADR 0011](0011-flashattention2-fp16-precision.md)

---

## Context

Module 4 adds four GPU operations (`attention_naive`, `rope`, `attention_prefill`, `attention_decode`) and their CPU reference counterparts (`attention`, `attention_online`, `rope`, `rope_precompute_tables`). Unlike earlier elementwise or 2-D matrix kernels, multi-head and grouped-query attention (MHA / GQA) spans both sequence tokens and attention heads, and bridges token-major linear projections with long-lived KV caches.

Locking the memory layouts, raw-pointer signatures, precision boundaries, and edge-case contracts before writing kernel code ensures zero-copy compatibility across prefill, decode, and Module 6 KV-cache slicing.

---

## Decision

### 1. Memory Layouts

| Tensor | Layout | Logical Shape | Element Offset Formula | Rationale |
|---|---|---|---|---|
| `Q`, `Out` | **Token-major** | `[s_q, n_heads, head_dim]` | `((pos * n_heads) + h) * head_dim + d` | Matches linear QKV/O projection row-major matrices (`[s_q, n_heads * head_dim]`) with zero transpose on both ends. At `s_q == 1` (decode), byte-identical to `[n_heads, head_dim]`. |
| `K`, `V` | **Head-major** | `[n_kv_heads, kv_max_seq_len, head_dim]` | `h_kv * kv_head_stride + t * head_dim + d` | Matches zero-copy axis-1 slices of a preallocated `[n_kv_heads, max_seq_len, head_dim]` KV cache (Module 1 `slice` + Module 6), and keeps `t` contiguous per head for coalesced KV sweeps. |
| `cos_table`, `sin_table` | **Row-major** | `[max_seq_len, head_dim / 2]` | `pos * (head_dim / 2) + i` | Shared across all heads; `pos` row fits inside L1/read-only cache (`<= 256 B` per token for `head_dim = 128`). |
| `out_lse` (optional) | **Token-major** | `[s_q, n_heads]` | `pos * n_heads + h` | Matches `Out` leading indexing; stores FP32 $\log \sum_j \exp(s_{i,j})$. |

**Contract on `kv_head_stride`:**
- Measured in elements (`half`), NOT bytes.
- Must satisfy `kv_head_stride >= s_kv * head_dim` when `s_kv > 0` (and `kv_head_stride >= 0` when `s_kv == 0`).
- Tightly packed contiguous `[n_kv_heads, s_kv, head_dim]` passes `kv_head_stride = s_kv * head_dim`; a preallocated KV cache passes `kv_head_stride = max_seq_len * head_dim`.
- Between `Q`/`K`/`V` heads and KV cache appends during Module 4 integration tests, `Tensor::permute({1, 0, 2}).contiguous()` or token-scatter populates the head-major KV cache (at `s == 1`, `[1, n_kv_heads, D]` and `[n_kv_heads, 1, D]` with `kv_head_stride = D` are already identical in memory).

### 2. Raw-Pointer Kernel APIs (`include/engine/kernels.hpp` & `include/engine/cpu_ref.hpp`)

Following [ADR 0003](0003-raw-pointer-kernel-api.md), launchers take raw pointers and explicit dimensions:

```cpp
enum class RopeConvention : std::uint8_t {
  RotateHalf = 0,    // Pairs (x[d], x[d + D/2]) -- Llama / HuggingFace convention
  AdjacentPair = 1,  // Pairs (x[2*d], x[2*d + 1]) -- Su et al. / GGUF convention
};

// Prefill / general MHA+GQA attention (Naive baseline & FlashAttention-2)
void attention_naive(const half* q, const half* k, const half* v, half* out,
                     std::int64_t s_q, std::int64_t s_kv,
                     std::int64_t n_heads, std::int64_t n_kv_heads,
                     std::int64_t head_dim, std::int64_t kv_head_stride,
                     float scale, std::int64_t q_pos_offset = 0,
                     bool is_causal = true, float* out_lse = nullptr,
                     cudaStream_t stream = 0);

void attention_prefill(const half* q, const half* k, const half* v, half* out,
                       std::int64_t s_q, std::int64_t s_kv,
                       std::int64_t n_heads, std::int64_t n_kv_heads,
                       std::int64_t head_dim, std::int64_t kv_head_stride,
                       float scale, std::int64_t q_pos_offset = 0,
                       bool is_causal = true, float* out_lse = nullptr,
                       cudaStream_t stream = 0);

// Decode MHA+GQA attention (s_q == 1 over 0..kv_len-1 keys)
// split_kv == 0 uses ADR-0013 auto heuristic; split_kv == 1 forces single-pass;
// split_kv > 1 forces explicit split count (clamped safely to kv_len / tile rules).
void attention_decode(const half* q, const half* k, const half* v, half* out,
                      std::int64_t kv_len, std::int64_t n_heads,
                      std::int64_t n_kv_heads, std::int64_t head_dim,
                      std::int64_t kv_head_stride, float scale,
                      int split_kv = 0, cudaStream_t stream = 0);

// In-place Rotary Position Embedding on token-major Q [s, n_heads, D]
// and K [s, n_kv_heads, D] (k may be nullptr iff n_kv_heads == 0).
void rope(half* q, half* k, const float* cos_table, const float* sin_table,
          std::int64_t s, std::int64_t n_heads, std::int64_t n_kv_heads,
          std::int64_t head_dim, std::int64_t pos_offset = 0,
          RopeConvention convention = RopeConvention::RotateHalf,
          cudaStream_t stream = 0);
```

CPU equivalents live under `engine::cpu::` using `engine::half` (`attention`, `attention_online`, `rope`, `rope_precompute_tables`).

### 3. Precision & Accumulation Rules

- **Attention inputs/outputs (`q`, `k`, `v`, `out`):** FP16 (`__half` / `engine::half`), per [ADR 0011](0011-flashattention2-fp16-precision.md).
- **Attention scale (`scale`):** Passed explicitly as `float` (typically `1.0f / sqrtf(float(head_dim))`). All `QK^T` dot products, masking, `max`, `expf`, running denominator `l`, `PV` output accumulators, and `out_lse` are **FP32**; conversion to FP16 occurs only once when writing `out`.
- **RoPE cos/sin tables:** Precomputed on host in **`double`** (`freq = std::pow(static_cast<double>(theta), -2.0 * i / head_dim)`, `angle = pos * freq`), stored in **FP32** (`[max_seq_len, head_dim / 2]`). Storing FP32 avoids phase-quantization drift beyond `pos >= 2048` where FP16 ulp exceeds angular resolution, while keeping table size tiny (`2048 * 64 * 4 * 2 = 1.0 MiB` at `D=128`). RoPE rotations execute in FP32 and store back to FP16.

### 4. Dimension, GQA, Causal, and Degenerate Input Contracts

Every launcher validates with `ENGINE_CHECK` (throwing `EngineError` before kernel launch):
1. Supported `head_dim`: `head_dim == 64 || head_dim == 128` (templated kernel instantiations). Other values throw `EngineError`.
2. GQA divisibility: `n_heads > 0`, `n_kv_heads > 0`, and `n_heads % n_kv_heads == 0`. GQA group size `group = n_heads / n_kv_heads`, mapping `kv_head = q_head / group`. Non-divisible ratios throw `EngineError`.
3. Non-negative dimensions/offsets: `s_q >= 0`, `s_kv >= 0` (or `kv_len >= 0`), `q_pos_offset >= 0`, `kv_head_stride >= (s_kv > 0 ? s_kv * head_dim : 0)`, finite `scale`, and `split_kv >= 0` for `attention_decode`.
4. Non-null pointers: `q`, `k`, `v`, `out` must be non-null (unless zero-element early exit contract allows, consistent with other kernels: null pointer checks execute first, zero-length early returns follow). For `rope`, `k != nullptr` whenever `n_kv_heads > 0`.
5. **Zero sequence lengths:**
   - `s_q == 0` (`s == 0` in `rope`): legal no-op (no kernel launch, no writes).
   - `s_kv == 0` with `s_q > 0` (`kv_len == 0` in decode): empty KV sequence represents zero attended context -> asynchronously zero-fills `out` (`cudaMemsetAsync`) and fills `out_lse` (if non-null) with `-infinity`, matching the `K == 0` zero-matrix contract in `matmul`/`gemv`.
6. **Causal masking with `q_pos_offset` and empty masked rows:**
   - When `is_causal == true`, query index `i` (`0 <= i < s_q`) at global position `q_global = q_pos_offset + i` attends strictly to keys `0 <= j < s_kv` satisfying `j <= q_global`. Keys `j > q_global` receive `-infinity` score prior to softmax.
   - If zero keys satisfy the causal condition (e.g., if all `j` are masked when `s_kv > 0` and all scores evaluate to `-infinity`, impossible for `q_pos_offset >= 0` except empty `s_kv == 0`, or future generalized masks), the output row is defined as `0.0` (not `NaN` from `0 / 0`) and `out_lse` is `-infinity`.
7. **Workspace buffers:**
   - Transient GPU scratch (`attention_naive` `[n_heads, s_q, s_kv]` FP32 scores buffer; `attention_decode` split-KV partials `[n_heads, splits, head_dim]` + LSE `[n_heads, splits, 2]`) uses `engine::ScopedPoolBuffer(engine::pool_allocator(engine::Device::CUDA), bytes, stream)` so all allocations are RAII-managed and stream-ordered without driver `cudaMalloc`/`cudaFree` overhead.

---

## Alternatives Considered and Rejected

1. **Head-major `[H_q, S, D]` for `Q` and `Out`:** Rejected because linear QKV projections emit `[S, H_q * D]` and output projection expects `[S, H_q * D]`. Making `Q`/`Out` head-major would force extra DRAM transpose passes before and after attention on every prefill layer.
2. **Token-major `[S, H_kv, D]` for `K` and `V`:** Rejected because KV caches grow token-by-token inside a fixed `[H_kv, max_seq_len, D]` buffer. Head-major layout makes `0..s_kv` a contiguous prefix slice (`stride(1) = D`) per KV head and prepares directly for Module 6 zero-copy `Tensor::slice`.
3. **Runtime `double`/FP16 sin/cos inside `rope` or recomputing trig per element:** Rejected — FP16 cos/sin loses precision at `pos >= 2048`, while device `sincosf` per element adds transcendental ALU cost when a tiny L1-resident FP32 lookup table works for both prefill and decode.
4. **Accepting `Tensor` objects instead of raw pointers:** Rejected per [ADR 0003](0003-raw-pointer-kernel-api.md) to keep the kernel layer standalone and independently testable.
