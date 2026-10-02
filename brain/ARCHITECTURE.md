# Architecture

> High-level view of the engine's structure, module boundaries, and data flow.
> Consult this when you need to understand how pieces fit together.

## Repository layout

```
bare-metal-llm-engine/
├── brain/              # AI context layer — persistent project memory (this folder)
├── docs/               # Technical documentation (human-readable)
│   ├── adr/            # Architectural Decision Records (ADR-0001 through ADR-0010)
│   ├── 01-dev-environment.md
│   ├── 02-cuda-exercises.md
│   ├── lab-notebook.md
│   ├── negative-results.md # Empirical boundaries & non-optimizations
│   └── roofline.png    # Hardware roofline plot
├── runbook/            # Operational procedures — step-by-step how-tos
├── include/engine/     # Public C++ headers
│   ├── dtype.hpp       # DType enum (F32, F16, BF16, I8, I4)
│   ├── half.hpp        # IEEE-754 binary16 bit-exact host representation & conversions
│   ├── tensor.hpp      # Tensor + Storage abstractions (Module 1)
│   ├── allocator.hpp   # Memory allocation interface
│   ├── pool_allocator.hpp # High-throughput PoolAllocator (slab + power-of-two buckets)
│   ├── cuda_stream.hpp # RAII CudaStream and CudaEvent primitives
│   ├── device_buffer.hpp  # RAII CUDA device buffer
│   ├── check.hpp       # Error handling macros (ENGINE_CHECK, CUDA_CHECK)
│   ├── cpu_ref.hpp     # CPU oracle function declarations
│   ├── kernels.hpp     # CUDA kernel launcher declarations
│   └── cuda_device.hpp # GPU device query utilities & live NVML clocks
├── src/                # C++ implementation
│   ├── tensor.cpp      # Tensor + Storage implementation
│   ├── allocator.cpp   # Default memory allocators
│   ├── pool_allocator.cpp # PoolAllocator implementation
│   ├── cuda_stream.cpp # RAII Stream & Event implementations
│   ├── dtype.cpp       # DType utilities
│   ├── cuda_device.cpp # GPU device queries & dynamic NVML loader
│   └── cpu_ref/        # CPU reference implementations (the oracle)
│       ├── vector_add_cpu.cpp
│       ├── reduce_cpu.cpp
│       ├── softmax_cpu.cpp
│       ├── rmsnorm_cpu.cpp
│       ├── matmul_cpu.cpp
│       ├── gemv_cpu.cpp
│       ├── rmsnorm_linear_cpu.cpp
│       ├── residual_rmsnorm_cpu.cpp
│       ├── embedding_cpu.cpp
│       ├── argmax_cpu.cpp
│       └── swiglu_cpu.cpp
├── kernels/            # CUDA kernels (the exercise ladder + Phase 4 optimizations)
│   ├── vector_add.cu   # Exercise 1 — WORKED EXAMPLE ✅
│   ├── reduce_sum.cu   # Exercise 2 — ✅ implemented (two-stage + warp shuffle)
│   ├── softmax.cu      # Exercise 3 — ✅ implemented (three-pass + block reduction)
│   ├── rmsnorm.cu      # Exercise 4 — ✅ implemented (sum-of-squares + rsqrtf)
│   ├── matmul_naive.cu # Exercise 5 — ✅ implemented (coalesced 16x16 2D mapping)
│   ├── matmul_tiled.cu # Exercise 6 — ✅ implemented (shared-memory 32x32 tiled GEMM)
│   ├── matmul_register_tiled.cu # Phase 4 — ✅ 2D register-tiled GEMM (128x128, 8x8 tile, float4)
│   ├── gemv.cu         # Exercise 7 — ✅ implemented (M=1 decode, 64-col tile, float2)
│   ├── gemv_fp16.cu    # Phase 3 — ✅ implemented (M=1 decode FP16, half2, float acc)
│   ├── embedding.cu    # Phase 3 — ✅ implemented (128-bit vector gather)
│   ├── argmax.cu       # Phase 3 — ✅ implemented (16-warp shuffle reduction)
│   ├── swiglu.cu       # Phase 4 — ✅ implemented (fused SiLU + Mul, float4/uint4)
│   ├── rmsnorm_linear.cu # Exercise 8 — ✅ implemented (fused RMSNorm + GEMM projection)
│   └── residual_rmsnorm.cu # Exercise 9 — ✅ implemented (fused Residual Add + RMSNorm)
├── tests/              # Test suites (119 tests total across 8 suites, 100% passing)
│   ├── test_framework.hpp/cpp  # Custom test harness with TEST/TEST_PENDING
│   ├── test_dtype.cpp          # DType tests
│   ├── test_tensor.cpp         # Module 1 specification (tensor tests)
│   ├── test_allocator.cpp      # Phase 4 PoolAllocator tests (100k cycles acceptance)
│   ├── test_cpu_ref.cpp        # CPU oracle tests (25 tests)
│   ├── test_kernels.cu         # CUDA kernel tests (45 tests)
│   ├── golden.hpp/cpp          # Golden file loader
│   └── golden/                 # Float64 reference data (gitignored, regenerated)
├── bench/              # Benchmark harness + benchmarks
│   ├── bench_harness.hpp       # CUDA-event timing, warmup, JSON export, NVML clocks
│   ├── bench_cpu_ref.cpp       # CPU reference baselines (--json support)
│   └── bench_kernels.cu        # CUDA kernel benchmarks (--json support)
├── tools/
│   ├── gen_reference.py        # Float64 reference data generator (PyTorch/NumPy)
│   ├── generate_results_table.py # Automated README benchmark table updater
│   ├── roofline_plot.py        # Roofline model generator (produces docs/roofline.png)
│   └── run_llama_bench.sh      # External llama.cpp benchmark runner
├── cmake/              # CMake modules
│   ├── EngineCuda.cmake        # Multi-arch CUDA detection (86;89)
│   ├── EngineWarnings.cmake    # Warning flags
│   └── engine_config.hpp.in    # Config header template (injects git hash & build info)
├── scripts/            # Build/test/run scripts
│   ├── build.ps1 / build.sh
│   ├── test.sh
│   ├── gpu-run.sh
│   └── profile_kernels.sh      # Nsight Compute (ncu) profiling script
├── CMakeLists.txt      # Root build configuration (ENGINE_CUDA_ARCH="86;89")
├── .clang-format       # Code formatting rules
├── .gitignore
└── README.md           # Human-readable project overview & auto-synced benchmark results
```

## Module structure

### Module 1 — Tensor Library (Objective 1, Jul – Aug 2026)

The foundation. Two types with a deliberate separation (ADR 0002):

```
Storage       — reference-counted ownership of a flat byte buffer on one device
  └── Tensor  — a VIEW onto a Storage: shape, strides, byte offset, dtype
```

Multiple Tensors can share one Storage. This is required for:
- KV-cache slices (Feb 2027): `cache[:, :pos, :]` must be a zero-copy view
- Weight tensors from mmap'd GGUF files (Mar 2027): ~50 tensors alias one region
- Reshapes in attention: `[B, S, H*D]` → `[B, S, H, D]` must be free

**Status:** Implemented in `src/tensor.cpp` (all 10 steps). Tests written as `TEST()`
in `tests/test_tensor.cpp`.

### Module 2 — CUDA Kernel Ladder & Decode GEMV (Objective 2, Aug – Sep 2026)

Seven foundational kernels, progressing from 1D bandwidth to 2D shared memory and decode GEMV:

| # | Kernel | Arithmetic Intensity | Status | Achieved Performance (RTX 4070 SUPER) |
|---|---|---|---|---|
| 1 | `vector_add` | 0.08 (memory-bound) | ✅ Complete | 432.6 GB/s (85.8% peak BW) |
| 2 | `reduce_sum` | 0.25 (memory-bound) | ✅ Complete | 458.3 GB/s (90.9% peak BW) |
| 3 | `softmax_rows` | 0.60 (memory-bound) | ✅ Complete | 435.7 GB/s (86.4% peak BW) |
| 4 | `rmsnorm` | 0.50 (memory-bound) | ✅ Complete | 435.5 GB/s (86.4% peak BW) |
| 5 | `matmul_naive` | 0.25 (memory-bound) | ✅ Complete | 1,852 GFLOP/s @ 4096³ |
| 6 | `matmul_tiled` | 8.00 (memory-bound) | ✅ Complete | 2,563 GFLOP/s @ 4096³ |
| 7 | `gemv` | 0.25 (memory-bound, $M=1$) | ✅ Complete | **473.5 GB/s (94.0% peak BW)** · 13.7% faster than cuBLAS |

Balance point on RTX 4070 SUPER: ~70 FLOP/byte (RTX 4090: ~82 FLOP/byte). All decoding ops are memory-bound.

### Module 3 — Kernel Fusion (Objective 3, Sep – Oct 2026)

Two fused operations targeting the memory wall between sub-layers (ADR 0006):

| # | Kernel | Target | Status | Architectural Impact |
|---|---|---|---|---|
| 8 | `rmsnorm_linear` | Pre-attention / Pre-FFN | ✅ Complete | Keeps normalized activation in shared memory (17.2 KiB); eliminates 16 MiB DRAM roundtrip at $M=512$ |
| 9 | `residual_rmsnorm` | Post-attention / Post-FFN | ✅ Complete | Single-pass residual accumulation + RMSNorm; eliminates 8 MiB DRAM read traffic |

- **Decode ($M=1$) vs Prefill ($M=512$) Fusion Dynamics:**
  - At $M=1$, intermediate activation tensor is only 16 KiB ($1 \times 4096 \times 4$ bytes), which fits comfortably in the 48 MiB hardware L2 cache. Therefore, fusion speedup at decode shape is driven by kernel launch overhead elimination (~3–5 µs saved per projection) rather than DRAM traffic reduction.
  - At prefill shapes ($M \ge 512$), intermediate tensors spill out of cache, making shared-memory fusion deliver substantial DRAM bandwidth reductions.

### Phase 4 — Production Foundation & 2D Register Tiling (Oct 2026)

Advanced high-throughput GEMM, memory management, and activation fusion:

| # | Component / Kernel | Target | Status | Architectural Impact |
|---|---|---|---|---|
| 10 | `gemv_fp16` | Token generation in FP16 | ✅ Complete | Coalesced `half2` vector loads, FP32 accumulator. Achieves **0.076 ms** ($1.89\times$ speedup over FP32) |
| 11 | `embedding` | Token ID gather | ✅ Complete | 128-bit vector instructions (`float4`, `uint4`). $8.9\ \mu\text{s}$ execution time |
| 12 | `argmax` | Greedy decoding | ✅ Complete | 16-warp shuffle reduction with deterministic lowest-index tie breaking. $10\ \mu\text{s}$ latency |
| 13 | `matmul_register_tiled` | Compute-bound GEMM | ✅ Complete | $128 \times 128$ block, $8 \times 8$ register tile, transposed `s_A`, `float4` loads. Achieves **16.17–17.55 TFLOP/s** ($6.7\times$ over `matmul_tiled`, **73.0% of cuBLAS**) |
| 14 | `swiglu` | Fused MLP activation | ✅ Complete | Single-pass $\text{SiLU}(\text{gate}) \cdot \text{up}$ ($40\%$ DRAM traffic reduction). **0.295 ms** ($1.52\times$ over unfused) |
| -- | `PoolAllocator` | Host & Device memory | ✅ Complete | Slab pre-allocation, power-of-two size classes, 256-byte alignment. Passed 100k cycles test with 1 driver alloc |
| -- | `CudaStream` / `CudaEvent` | Asynchronous compute | ✅ Complete | Move-only RAII wrappers with non-blocking flags and safe synchronization |

### Module 4 — FlashAttention-2 & Core Attention Architecture (Oct – Nov 2026)
- **Target:** Fused Multi-Head Attention forward pass without materializing the $S \times S$ attention matrix, reducing DRAM memory from $O(S^2)$ to $O(S)$.
- **Components:**
  - **Rotary Position Embeddings (RoPE):** Pairwise 2D rotations applied to $Q$ and $K$ heads before attention dot product (`kernels/rope.cu`).
  - **Naive Attention Baseline Oracle:** $S \times S$ materialized reference checking causal masking and GQA (`src/cpu_ref/attention_cpu.cpp`, `kernels/attention_naive.cu`).
  - **Prefill FlashAttention-2 Kernel:** Tiled $Q K^T$ in shared memory ($B_r \times B_c$), online softmax rescale loop ($m_{\text{new}}, \ell_{\text{new}}$), register-accumulated $P V$ projection, and causal mask branch skipping (`kernels/flash_attention.cu`).
  - **Decode Specialization (FlashDecoding):** Split-KV reduction kernel for $S_q=1$ autoregressive decoding across extended context windows ($S_{kv} \in [1, 2048]$).
  - **Grouped-Query Attention (GQA):** $H_Q / H_{KV}$ query-to-KV head mapping (e.g. 32 Q heads sharing 4 KV heads for TinyLlama-1.1B).

### Module 5 — Weight-Only Quantization (Nov – Dec 2026)
- **Target:** Sub-byte weight storage and high-throughput on-the-fly dequantization.
- **Components:**
  - INT4 / INT8 packed storage schemes (AWQ / GPTQ layout compatibility).
  - Fast bit-unpacking SIMD / PTX intrinsics (`lop3.b32`, `prmt`).
  - Fused dequantize-GEMV kernel for $M=1$ token generation.
  - Verification harness comparing against FP32 unquantized oracles.

### Module 6 — KV-Cache & Attention Memory Optimization (Jan – Feb 2027)
- **Target:** Constant-memory KV storage across extended context lengths.
- **Components:**
  - PagedAttention / segmented circular buffer allocation.
  - Zero-copy slice views utilizing Module 1 `Storage`/`Tensor` abstractions.
  - Rotary Position Embeddings (RoPE) fused into attention projection.

### Module 7 — GGUF Model Pipeline & End-to-End Inference (Mar – Apr 2027)
- **Target:** Complete autoregressive text generation from disk weights.
- **Components:**
  - GGUF v3 file parser with zero-copy `mmap` backing.
  - Byte-Pair Encoding (BPE) tokenizer implementation.
  - Temperature, top-k, and top-p (nucleus) samplers.
  - Autoregressive generation loop with TinyLlama-1.1B and LLaMA-2-7B.

### Module 8 — Bindings, llama-bench Comparison & Thesis (May – Jun 2027)
- **Target:** Production readiness, external validation, and formal documentation.
- **Components:**
  - Python bindings via `pybind11` for high-level interaction.
  - Comparative benchmark evaluation against `llama.cpp` (`llama-bench`).
  - Final thesis report, complete roofline visualizations, and negative results analysis.

## Three-tier correctness model

```
Tier 1: tools/gen_reference.py
        PyTorch/NumPy, float64, on the host
        Writes binary golden files to tests/golden/
           │
           ├──compared──▶ Tier 2: src/cpu_ref/*.cpp
           │                      Scalar C++ oracle, checked against Tier 1
           │
           └──compared──▶ Tier 3: kernels/*.cu
                                  CUDA kernels, checked against Tier 1 (NOT Tier 2)
```

**Why Tier 3 is NOT compared to Tier 2:** A shared misunderstanding (transposed layout,
off-by-one in row indexing) would pass if both were compared to each other. Comparing
each independently against an external float64 answer means a shared misconception has
to survive two implementations and a third-party numerical reference.

## Data flow (inference, future)

```
GGUF file on disk
    │ mmap
    ▼
Storage (non-owning, BorrowTag) ──▶ ~50 weight Tensors (views)
    │
    ▼
Tokenizer → token IDs
    │
    ▼
Embedding lookup → [1, S, D] Tensor
    │
    ▼
┌─── Per transformer layer (×32 for Llama-7B) ───┐
│ RMSNorm                                         │
│ Multi-Head Attention (QKV projection, FlashAttn) │
│ RMSNorm                                         │
│ Feed-Forward (gate, up, down projections)        │
└──────────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → Output projection → logits [1, V]
    │
    ▼
Softmax → Sampler → next token ID
    │
    ▼
Detokenize → text output
```
