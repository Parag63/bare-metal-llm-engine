# ADR 0007 — Dedicated GEMV Kernel for Autoregressive Token Generation

**Status:** accepted · **Date:** 2026-10-01 · **Applies to:** `kernels/gemv.cu`, `src/cpu_ref/gemv_cpu.cpp`, `bench/bench_kernels.cu`

## Context

In autoregressive transformer generation (decoding phase), every generated token executes linear projections with batch/sequence dimension $M=1$:
- Query, Key, and Value projections: $1 \times K \to 1 \times D$
- Feed-Forward gate, up, and down projections: $1 \times K \to 1 \times 4K$ or $1 \times 4K \to 1 \times K$
- Output projection to vocabulary logits: $1 \times K \to 1 \times V$

When $M=1$, the general matrix multiplication ($C_{M \times N} = A_{M \times K} B_{K \times N}$) degenerates into a matrix-vector product (GEMV):
$$y = x \cdot B$$
where $x$ is a $1 \times K$ vector, $B$ is a $K \times N$ weight matrix, and $y$ is a $1 \times N$ vector.

In standard 2D tiled GEMM (such as `matmul_tiled` with $32 \times 32$ or $16 \times 16$ tiles):
1. A 2D grid assigns $32 \times 32$ thread blocks. Because $M=1$, 31 out of 32 rows in every thread block are idle ($threadIdx.y > 0$).
2. The arithmetic intensity of GEMV is strictly memory-bound:
   $$\text{AI} = \frac{2 \cdot K \cdot N \text{ FLOP}}{4 \cdot K + 4 \cdot K \cdot N + 4 \cdot N \text{ bytes}} \approx \frac{2}{4} = 0.50 \text{ FLOP/byte (or } 0.25 \text{ if counting ideal input+output)}$$
   On the RTX 4070 SUPER (504 GB/s, 35.5 TFLOP/s FP32), the memory balance point is ~70 FLOP/byte. At 0.25–0.50 FLOP/byte, execution is >100× below the balance point.
3. Benchmarking `matmul_tiled` at $1 \times 4096 \times 4096$ yields only **128.5 GB/s (25.5% peak bandwidth)**, wasting ~75% of available hardware throughput.

## Decision

We implement a dedicated, specialized GEMV kernel (`kernels/gemv.cu`) optimized strictly for $M=1$:

1. **1D Column Block Tiling:**
   - Each thread block processes a contiguous tile of columns: `COLS_PER_BLOCK = 64`.
   - Grid dimension is configured as `grid_dim.x = (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK`.
   - For LLaMA-7B projection shapes ($N = 4096, 11008, 12288$), this saturates the GPU SMs with 64 to 192 thread blocks.

2. **Cooperative Reduction over $K$:**
   - Each block employs 256 threads grouped into 8 warps (`WARPS_PER_BLOCK = 8`, `COLS_PER_WARP = 8`).
   - Each warp is assigned 8 specific columns. The 32 threads in the warp divide the $K$ dimension, each striding across $K$ with step size 32.
   - Reduction within the warp is executed using zero-overhead `__shfl_down_sync` register tree shuffles across lane offsets 16, 8, 4, 2, 1.

3. **128-bit / 64-bit Coalesced Memory Access:**
   - Vector elements and matrix entries are loaded using `float2` (64-bit) vectorized transactions, halving instruction count and maximizing memory pipeline saturation.

4. **Raw Pointer Launcher Interface:**
   - Conforms strictly to ADR 0003, accepting raw device pointers `const float* A`, `const float* B`, `float* C`, `int64_t K`, `int64_t N`, and an optional `cudaStream_t`.

## Consequences

- **Performance vs 2D Tiled GEMM:**
  - At $1 \times 4096 \times 4096$: Achieves **465.4 GB/s (92.3% peak bandwidth)**, a **$3.62\times$ speedup** over `matmul_tiled` (128.5 GB/s).
  - At $1 \times 12288 \times 4096$: Achieves **473.5 GB/s (94.0% peak bandwidth)**.
- **Performance vs cuBLAS:**
  - cuBLAS (`cublasSgemm` / `cublasSgemv`) reaches 416.3 GB/s on $1 \times 12288 \times 4096$. Our hand-written GEMV is **13.7% faster than cuBLAS** due to zero-overhead launch parameters and specialized register tile allocation.
- **Verification:**
  - Verified against both scalar CPU oracle (`src/cpu_ref/gemv_cpu.cpp`) and PyTorch/NumPy float64 golden reference data (`tests/test_kernels.cu`).
- **Integration:**
  - Forms the foundation for the upcoming Module 5 quantized GEMV kernel (weight-only INT4/INT8 dequantization).
