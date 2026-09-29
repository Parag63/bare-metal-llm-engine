# Architecture

> High-level view of the engine's structure, module boundaries, and data flow.
> Consult this when you need to understand how pieces fit together.

## Repository layout

```
bare-metal-llm-engine/
├── brain/              # AI context layer — persistent project memory (this folder)
├── docs/               # Technical documentation (human-readable)
│   ├── adr/            # Architectural Decision Records (ADR-0001 through ADR-0005)
│   ├── 01-dev-environment.md
│   ├── 02-cuda-exercises.md
│   └── lab-notebook.md
├── runbook/            # Operational procedures — step-by-step how-tos
├── include/engine/     # Public C++ headers
│   ├── dtype.hpp       # DType enum (F32, F16, BF16, I8, I4)
│   ├── tensor.hpp      # Tensor + Storage abstractions (Module 1)
│   ├── allocator.hpp   # Memory allocation interface
│   ├── device_buffer.hpp  # RAII CUDA device buffer
│   ├── check.hpp       # Error handling macros (ENGINE_CHECK, CUDA_CHECK)
│   ├── cpu_ref.hpp     # CPU oracle function declarations
│   ├── kernels.hpp     # CUDA kernel launcher declarations
│   └── cuda_device.hpp # GPU device query utilities
├── src/                # C++ implementation
│   ├── tensor.cpp      # Tensor + Storage implementation (559 lines, Module 1)
│   ├── allocator.cpp   # Memory allocator
│   ├── dtype.cpp       # DType utilities
│   ├── cuda_device.cpp # GPU device queries
│   └── cpu_ref/        # CPU reference implementations (the oracle)
│       ├── vector_add_cpu.cpp
│       ├── reduce_cpu.cpp
│       ├── softmax_cpu.cpp
│       ├── rmsnorm_cpu.cpp
│       └── matmul_cpu.cpp
├── kernels/            # CUDA kernels (the exercise ladder)
│   ├── vector_add.cu   # Exercise 1 — WORKED EXAMPLE ✅
│   ├── reduce_sum.cu   # Exercise 2 — ✅ implemented (two-stage + warp shuffle)
│   ├── softmax.cu      # Exercise 3 — ✅ implemented (three-pass + block reduction)
│   ├── rmsnorm.cu      # Exercise 4 — ✅ implemented (sum-of-squares + rsqrtf)
│   ├── matmul_naive.cu # Exercise 5 — ✅ implemented (coalesced 16x16 2D mapping)
│   └── matmul_tiled.cu # Exercise 6 — ✅ implemented (shared-memory 32x32 tiled GEMM)
├── tests/              # Test suites
│   ├── test_framework.hpp/cpp  # Custom test harness with TEST/TEST_PENDING
│   ├── test_dtype.cpp          # DType tests
│   ├── test_tensor.cpp         # Module 1 specification (tensor tests)
│   ├── test_cpu_ref.cpp        # CPU oracle tests
│   ├── test_kernels.cu         # CUDA kernel tests (exercise ladder)
│   ├── golden.hpp/cpp          # Golden file loader
│   └── golden/                 # Float64 reference data (gitignored, regenerated)
├── bench/              # Benchmark harness + benchmarks
│   ├── bench_harness.hpp       # CUDA-event timing, warmup, median/min reporting
│   ├── bench_cpu_ref.cpp       # CPU reference baselines
│   └── bench_kernels.cu        # CUDA kernel benchmarks
├── tools/
│   └── gen_reference.py        # Float64 reference data generator (PyTorch/NumPy)
├── cmake/              # CMake modules
│   ├── EngineCuda.cmake        # CUDA detection (never fatal)
│   ├── EngineWarnings.cmake    # Warning flags
│   └── engine_config.hpp.in    # Config header template
├── scripts/            # Build/test/run scripts
│   ├── build.ps1 / build.sh
│   ├── test.sh
│   └── gpu-run.sh
├── colab/              # (empty, future use)
├── CMakeLists.txt      # Root build configuration
├── .clang-format       # Code formatting rules
├── .gitignore
└── README.md           # Human-readable project overview
```

## Module structure

### Module 1 — Tensor Library (Objective 1, Oct 2026)

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

### Module 2 — CUDA Kernel Ladder (Objective 2)

Six kernels, each building on the previous:

| # | Kernel | Arithmetic Intensity | Status |
|---|---|---|---|
| 1 | `vector_add` | 0.08 (memory-bound) | ✅ Complete (432.6 GB/s, 85.8% peak) |
| 2 | `reduce_sum` | 0.25 (memory-bound) | ✅ Complete (458.3 GB/s, 90.9% peak) |
| 3 | `softmax_rows` | 0.6 (memory-bound) | ✅ Complete (435.7 GB/s, 86.4% peak) |
| 4 | `rmsnorm` | 0.5 (memory-bound) | ✅ Complete (435.5 GB/s, 86.4% peak) |
| 5 | `matmul_naive` | 0.25 (memory-bound) | ✅ Complete (1,852 GFLOP/s @ 4096³) |
| 6 | `matmul_tiled` | 8.0 (still memory-bound) | ✅ Complete (2,563 GFLOP/s @ 4096³) |

Balance point on RTX 4090: ~82 FLOP/byte. Everything below that is memory-bound.

### Module 3+ — Future Modules (not yet started)

- **Kernel fusion** (Dec 2026) — fuse elementwise ops to eliminate intermediate traffic
- **FlashAttention** (Jan–Feb 2027) — online softmax applied tile-by-tile
- **Quantization** — packed INT4 weights with dequantization kernels
- **KV-cache** — ring-buffer cache with zero-copy slicing
- **Model loader** — GGUF-style mmap'd weight loading
- **Tokenizer** — BPE tokenizer
- **Sampler** — top-k, top-p, temperature scaling
- **Python bindings** — for usability

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
