# LLM Inference on Bare Metal — Complete Flowchart

> **Purpose:** A detailed, industry-level reference diagram showing every stage of
> running a large language model on a single machine — from loading weights off disk
> to emitting the final token. This documents the *target architecture* of the
> bare-metal-llm-engine project, independent of what has been implemented so far.
>
> **Audience:** The project author, the mentor, and anyone reviewing the project who
> needs to understand the full inference pipeline at a glance.

---

## 1. End-to-End Inference Pipeline (Top Level)

This is the bird's-eye view. Each box expands into its own detailed sub-diagram below.

```mermaid
flowchart TB
    subgraph PHASE_0["Phase 0 — Startup"]
        A0["Load Model Config<br/>(architecture, dims, n_layers)"]
        A1["Memory-Map Weight File<br/>(GGUF / safetensors)"]
        A2["Allocate GPU Memory<br/>(cudaMalloc pools)"]
        A3["Transfer Weights to GPU<br/>(H2D via cudaMemcpyAsync)"]
        A4["Initialize KV Cache<br/>(pre-allocate max_seq_len slots)"]
        A0 --> A1 --> A2 --> A3 --> A4
    end

    subgraph PHASE_1["Phase 1 — Tokenization"]
        B0["Receive Raw Text Prompt"]
        B1["BPE Tokenization<br/>(byte-pair encoding)"]
        B2["Map Tokens to IDs<br/>(vocab lookup table)"]
        B3["Prepend BOS Token<br/>(beginning-of-sequence)"]
        B0 --> B1 --> B2 --> B3
    end

    subgraph PHASE_2["Phase 2 — Prefill"]
        C0["Embedding Lookup<br/>(token IDs → dense vectors)"]
        C1["Run All Transformer Layers<br/>(full sequence, parallel)"]
        C2["Populate KV Cache<br/>(store K,V for all positions)"]
        C3["Final RMSNorm"]
        C4["Logits = hidden × W_vocab^T"]
        C0 --> C1 --> C2 --> C3 --> C4
    end

    subgraph PHASE_3["Phase 3 — Autoregressive Decode"]
        D0["Embed Last Generated Token"]
        D1["Run All Transformer Layers<br/>(single token, read KV cache)"]
        D2["Append New K,V to Cache"]
        D3["Final RMSNorm"]
        D4["Logits = hidden × W_vocab^T"]
        D5["Sample Next Token<br/>(temperature / top-k / top-p)"]
        D6{"EOS or<br/>max_len?"}
        D0 --> D1 --> D2 --> D3 --> D4 --> D5 --> D6
        D6 -- "No" --> D0
    end

    subgraph PHASE_4["Phase 4 — Detokenization"]
        E0["Collect Generated Token IDs"]
        E1["Map IDs Back to Text<br/>(vocab reverse lookup)"]
        E2["Handle Byte-Level Merges<br/>(UTF-8 reassembly)"]
        E3["Emit Final Text Output"]
        E0 --> E1 --> E2 --> E3
    end

    PHASE_0 --> PHASE_1 --> PHASE_2 --> PHASE_3 --> PHASE_4
    D6 -- "Yes" --> E0

    style PHASE_0 fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style PHASE_1 fill:#16213e,stroke:#0f3460,color:#e0e0e0
    style PHASE_2 fill:#0f3460,stroke:#533483,color:#e0e0e0
    style PHASE_3 fill:#533483,stroke:#e94560,color:#e0e0e0
    style PHASE_4 fill:#e94560,stroke:#ffffff,color:#ffffff
```

---

## 2. Model Loading — Weight Format and GPU Transfer

```mermaid
flowchart LR
    subgraph DISK["Disk (SSD/NVMe)"]
        F0["Model File<br/>(GGUF / safetensors / .bin)"]
        F1["Metadata Header<br/>(architecture, vocab_size,<br/>hidden_dim, n_heads, n_layers,<br/>rope_theta, norm_eps)"]
        F2["Tensor Directory<br/>(name → offset, shape, dtype)"]
        F3["Raw Weight Data<br/>(FP16 / BF16 / INT4 / INT8)"]
        F0 --> F1
        F0 --> F2
        F0 --> F3
    end

    subgraph MMAP["Memory Mapping"]
        G0["mmap() the file<br/>(zero-copy, demand-paged)"]
        G1["OS Page Cache<br/>(kernel manages eviction)"]
        G2["Random Access<br/>(pointer arithmetic on tensors)"]
        G0 --> G1 --> G2
    end

    subgraph QUANT["Dequantization (if needed)"]
        H0{"Weight dtype?"}
        H1["FP16/BF16:<br/>Direct copy or cast to FP32"]
        H2["INT8:<br/>scale × int8_val per group"]
        H3["INT4 (Q4_K_M etc.):<br/>Unpack nibbles,<br/>scale + min per block of 32"]
        H0 -- "FP16/BF16" --> H1
        H0 -- "INT8" --> H2
        H0 -- "INT4" --> H3
    end

    subgraph GPU_MEM["GPU Memory Layout"]
        I0["Weight Buffers<br/>(one per layer, pinned layout)"]
        I1["KV Cache Pool<br/>(n_layers × 2 × max_seq × head_dim)"]
        I2["Activation Scratch<br/>(reused across layers)"]
        I3["Embedding Table<br/>(vocab_size × hidden_dim)"]
        I4["Output Projection<br/>(hidden_dim × vocab_size)"]
    end

    F3 --> G0
    G2 --> H0
    H1 --> I0
    H2 --> I0
    H3 --> I0

    style DISK fill:#2d2d44,stroke:#444,color:#e0e0e0
    style MMAP fill:#1e3a5f,stroke:#444,color:#e0e0e0
    style QUANT fill:#3a1e5f,stroke:#444,color:#e0e0e0
    style GPU_MEM fill:#5f1e3a,stroke:#444,color:#e0e0e0
```

---

## 3. BPE Tokenization Pipeline

```mermaid
flowchart TB
    T0["Raw Input String<br/>(UTF-8 bytes)"]
    T1["Pre-tokenization<br/>(split on whitespace / regex rules)"]
    T2["Byte-Level Encoding<br/>(each byte → base vocab token)"]
    T3["Iterative Merge Loop"]

    subgraph MERGE_LOOP["BPE Merge Loop"]
        M0["Find all adjacent pairs<br/>in current token sequence"]
        M1["Look up pair priorities<br/>in merge table (sorted by rank)"]
        M2{"Highest-priority<br/>pair found?"}
        M3["Merge the pair into<br/>a single new token"]
        M4["Update adjacency"]
        M0 --> M1 --> M2
        M2 -- "Yes" --> M3 --> M4 --> M0
    end

    T5["Final Token ID Sequence"]
    T6["Prepend BOS = 1"]
    T7["Output: int32 array<br/>ready for embedding lookup"]

    T0 --> T1 --> T2 --> T3 --> MERGE_LOOP
    M2 -- "No (converged)" --> T5 --> T6 --> T7

    style MERGE_LOOP fill:#1a3a2e,stroke:#2d5a3e,color:#e0e0e0
```

---

## 4. Single Transformer Layer (The Core Compute)

This is where 99%+ of the FLOPs happen. A LLaMA-style layer is shown.

```mermaid
flowchart TB
    IN["Input Hidden State<br/>x ∈ R^(seq × hidden_dim)"]

    subgraph ATTN_BLOCK["Self-Attention Sub-Layer"]
        direction TB
        N1["RMSNorm(x)<br/>x_norm = x / sqrt(mean(x²) + eps) × γ"]

        subgraph QKV["QKV Projection (3 GEMMs or 1 fused)"]
            Q["Q = x_norm × W_Q<br/>(seq × n_heads × head_dim)"]
            K["K = x_norm × W_K<br/>(seq × n_kv_heads × head_dim)"]
            V["V = x_norm × W_V<br/>(seq × n_kv_heads × head_dim)"]
        end

        subgraph ROPE["Rotary Position Embedding (RoPE)"]
            R1["Compute sin/cos tables<br/>(position × head_dim/2)"]
            R2["Rotate Q pairs:<br/>(q0,q1) → (q0·cos - q1·sin,<br/>q0·sin + q1·cos)"]
            R3["Rotate K pairs:<br/>(same rotation as Q)"]
        end

        subgraph GQA["Grouped-Query Attention"]
            KV_EXP["Expand K,V heads<br/>(if n_kv_heads < n_heads,<br/>broadcast to match Q heads)"]
            SC["Scores = Q × K^T / sqrt(head_dim)<br/>(n_heads × seq × seq)"]
            MASK["Apply Causal Mask<br/>(upper triangle = -inf)"]
            SM["Softmax(Scores)<br/>(row-wise, numerically stable)"]
            AV["Attention Output = Softmax × V<br/>(n_heads × seq × head_dim)"]
        end

        PROJ["Output Projection<br/>attn_out = concat(heads) × W_O"]

        N1 --> QKV
        Q --> ROPE
        K --> ROPE
        ROPE --> GQA
        V --> KV_EXP
        R2 --> SC
        R3 --> SC
        KV_EXP --> SC
        SC --> MASK --> SM --> AV
        KV_EXP --> AV
        AV --> PROJ
    end

    RES1["Residual Connection<br/>h = x + attn_out"]

    subgraph FFN_BLOCK["Feed-Forward Sub-Layer (SwiGLU)"]
        direction TB
        N2["RMSNorm(h)<br/>h_norm = h / sqrt(mean(h²) + eps) × γ"]

        subgraph SWIGLU["SwiGLU FFN"]
            GATE["Gate = h_norm × W_gate<br/>(seq × intermediate_dim)"]
            UP["Up = h_norm × W_up<br/>(seq × intermediate_dim)"]
            SILU["SiLU(Gate) = Gate × σ(Gate)<br/>(elementwise)"]
            MUL["Gated = SiLU(Gate) ⊙ Up<br/>(elementwise multiply)"]
            DOWN["FFN_out = Gated × W_down<br/>(seq × hidden_dim)"]
        end

        N2 --> SWIGLU
        GATE --> SILU
        UP --> MUL
        SILU --> MUL
        MUL --> DOWN
    end

    RES2["Residual Connection<br/>out = h + ffn_out"]

    IN --> ATTN_BLOCK
    ATTN_BLOCK --> RES1
    RES1 --> FFN_BLOCK
    FFN_BLOCK --> RES2

    style ATTN_BLOCK fill:#1e2a4a,stroke:#3a5a8a,color:#e0e0e0
    style FFN_BLOCK fill:#2a1e4a,stroke:#5a3a8a,color:#e0e0e0
    style QKV fill:#2a3a5a,stroke:#4a6a9a,color:#e0e0e0
    style ROPE fill:#3a2a5a,stroke:#6a4a9a,color:#e0e0e0
    style GQA fill:#2a4a3a,stroke:#4a8a6a,color:#e0e0e0
    style SWIGLU fill:#4a2a3a,stroke:#8a4a6a,color:#e0e0e0
```

---

## 5. KV Cache — Prefill vs Decode

```mermaid
flowchart TB
    subgraph PREFILL["Prefill Phase (seq_len = S tokens at once)"]
        P0["Compute Q, K, V for ALL<br/>S prompt positions in parallel"]
        P1["Score = Q_all × K_all^T<br/>(full S × S attention matrix)"]
        P2["Store K_all, V_all<br/>into cache slots 0..S-1"]
        P3["Output: hidden state<br/>at position S-1 only<br/>(used for first decode step)"]
        P0 --> P1 --> P2 --> P3
    end

    subgraph DECODE["Decode Phase (1 new token at a time)"]
        D0["Compute Q, K, V for<br/>position t (single row)"]
        D1["Append K_t, V_t to cache<br/>(slot t)"]
        D2["Score = Q_t × K_cached^T<br/>(1 × t attention vector)"]
        D3["Apply causal mask<br/>(trivial: 1 × t is already causal)"]
        D4["Softmax over t scores"]
        D5["Output = softmax × V_cached<br/>(1 × head_dim)"]
        D0 --> D1 --> D2 --> D3 --> D4 --> D5
    end

    subgraph CACHE_MEM["KV Cache Memory Layout"]
        CM0["Per layer: 2 tensors<br/>(K cache, V cache)"]
        CM1["Shape: (max_seq_len × n_kv_heads × head_dim)"]
        CM2["Total memory:<br/>n_layers × 2 × max_seq × n_kv_heads × head_dim × sizeof(dtype)"]
        CM3["Example: LLaMA-7B, 2048 ctx, FP16<br/>= 32 × 2 × 2048 × 32 × 128 × 2B<br/>= 1.0 GiB"]
        CM0 --> CM1 --> CM2 --> CM3
    end

    PREFILL --> DECODE
    DECODE -.->|"reads from"| CACHE_MEM
    PREFILL -.->|"writes to"| CACHE_MEM

    style PREFILL fill:#1a3a5a,stroke:#2a5a8a,color:#e0e0e0
    style DECODE fill:#3a1a5a,stroke:#5a2a8a,color:#e0e0e0
    style CACHE_MEM fill:#3a3a1a,stroke:#5a5a2a,color:#e0e0e0
```

---

## 6. Token Sampling Strategies

```mermaid
flowchart TB
    L0["Raw Logits<br/>(vocab_size floats)"]

    subgraph TEMP["Temperature Scaling"]
        T1["logits = logits / temperature"]
        T2["temperature = 0: argmax (greedy)"]
        T3["temperature = 1: unmodified"]
        T4["temperature > 1: flatter distribution"]
    end

    subgraph TOPK["Top-K Filtering"]
        K1["Sort logits descending"]
        K2["Keep only top K tokens"]
        K3["Set rest to -infinity"]
    end

    subgraph TOPP["Top-P (Nucleus) Filtering"]
        P1["Sort by probability descending"]
        P2["Compute cumulative sum"]
        P3["Remove tokens beyond<br/>cumsum > p threshold"]
    end

    subgraph SAMPLE["Final Sampling"]
        S1["Softmax over<br/>remaining logits"]
        S2["Draw from categorical<br/>distribution (multinomial)"]
        S3["Output: single token ID"]
    end

    subgraph SPECIAL["Special Handling"]
        SP1["Repetition Penalty:<br/>penalize recently generated tokens"]
        SP2["Frequency Penalty:<br/>reduce prob proportional<br/>to occurrence count"]
        SP3["Presence Penalty:<br/>flat penalty if token<br/>has appeared at all"]
    end

    L0 --> TEMP --> TOPK --> TOPP --> SAMPLE
    SPECIAL -.->|"applied before softmax"| SAMPLE

    style TEMP fill:#3a2a1a,stroke:#5a4a2a,color:#e0e0e0
    style TOPK fill:#2a3a1a,stroke:#4a5a2a,color:#e0e0e0
    style TOPP fill:#1a3a2a,stroke:#2a5a4a,color:#e0e0e0
    style SAMPLE fill:#1a2a3a,stroke:#2a4a5a,color:#e0e0e0
    style SPECIAL fill:#3a1a2a,stroke:#5a2a4a,color:#e0e0e0
```

---

## 7. GPU Kernel Execution — Memory Hierarchy

```mermaid
flowchart TB
    subgraph HOST["Host (CPU + System RAM)"]
        H0["Application Code<br/>(inference loop, sampling)"]
        H1["Weight File (mmap)"]
        H2["Tokenizer State"]
        H3["Output Buffer"]
    end

    subgraph PCIE["PCIe Bus (16 GB/s)"]
        P0["H2D: Weights, Token IDs"]
        P1["D2H: Logits, Generated IDs"]
    end

    subgraph GPU["GPU"]
        subgraph DRAM["Global Memory (HBM/GDDR6X)<br/>~504 GB/s bandwidth"]
            G0["Weight Matrices<br/>(read-only, largest consumer)"]
            G1["KV Cache<br/>(read-write, grows with context)"]
            G2["Activation Buffers<br/>(temporary, reused per layer)"]
        end

        subgraph L2["L2 Cache (~48 MiB on RTX 4070S)"]
            L2_0["Automatic hardware cache<br/>(no explicit management)"]
            L2_1["Catches reuse within<br/>a few milliseconds"]
        end

        subgraph SM["Streaming Multiprocessor (×56)"]
            subgraph SMEM["Shared Memory (up to 100 KiB)"]
                SM0["Tiled matrix blocks<br/>(loaded cooperatively by block)"]
                SM1["Reduction scratch space<br/>(block_reduce_sum, etc.)"]
                SM2["RMSNorm row cache<br/>(fused kernels)"]
            end

            subgraph REGS["Registers (65536 per SM)"]
                R0["Thread-local accumulators"]
                R1["Loop variables, pointers"]
                R2["Warp shuffle intermediates"]
            end

            subgraph WARPS["Warps (32 threads each)"]
                W0["Coalesced global memory loads<br/>(128-byte transactions)"]
                W1["Warp-level shuffle reductions<br/>(__shfl_down_sync)"]
                W2["Divergence-free inner loops"]
            end
        end
    end

    HOST --> PCIE --> DRAM
    DRAM --> L2 --> SM
    SM0 --> REGS
    WARPS --> SMEM

    style HOST fill:#2a2a3a,stroke:#4a4a5a,color:#e0e0e0
    style PCIE fill:#3a3a2a,stroke:#5a5a4a,color:#e0e0e0
    style GPU fill:#1a2a3a,stroke:#2a4a5a,color:#e0e0e0
    style DRAM fill:#2a3a4a,stroke:#4a5a6a,color:#e0e0e0
    style L2 fill:#3a4a2a,stroke:#5a6a4a,color:#e0e0e0
    style SM fill:#2a2a4a,stroke:#4a4a6a,color:#e0e0e0
    style SMEM fill:#3a3a5a,stroke:#5a5a7a,color:#e0e0e0
    style REGS fill:#4a3a3a,stroke:#6a5a5a,color:#e0e0e0
    style WARPS fill:#3a4a3a,stroke:#5a6a5a,color:#e0e0e0
```

---

## 8. Kernel Fusion Opportunities in the Transformer

```mermaid
flowchart LR
    subgraph UNFUSED["Unfused (Naive) Pipeline"]
        direction TB
        U1["RMSNorm Kernel<br/>Write norm_out to DRAM"]
        U2["QKV GEMM Kernel<br/>Read norm_out from DRAM"]
        U3["RoPE Kernel<br/>Read/write Q,K to DRAM"]
        U4["Attention Score Kernel<br/>Write S matrix to DRAM"]
        U5["Softmax Kernel<br/>Read S, write P to DRAM"]
        U6["Attention Value Kernel<br/>Read P,V, write attn_out"]
        U7["Output Projection Kernel"]
        U8["Residual Add Kernel<br/>Write sum to DRAM"]
        U9["RMSNorm Kernel<br/>Read sum from DRAM"]
        U10["Gate GEMM Kernel"]
        U11["Up GEMM Kernel"]
        U12["SiLU Kernel"]
        U13["Elementwise Multiply Kernel"]
        U14["Down GEMM Kernel"]
        U15["Residual Add Kernel"]
        U1 --> U2 --> U3 --> U4 --> U5 --> U6 --> U7 --> U8 --> U9 --> U10 --> U11 --> U12 --> U13 --> U14 --> U15
    end

    subgraph FUSED["Fused (Optimized) Pipeline"]
        direction TB
        F1["Fused RMSNorm + QKV Proj<br/>(Exercise 7: rmsnorm_linear)"]
        F2["Fused RoPE + Score + Mask<br/>+ Softmax + Value Multiply<br/>(FlashAttention)"]
        F3["Output Projection GEMM"]
        F4["Fused Residual + RMSNorm<br/>(Exercise 8: residual_rmsnorm)"]
        F5["Fused Gate + Up + SiLU + Mul<br/>(SwiGLU fusion)"]
        F6["Down Projection GEMM"]
        F7["Fused Residual + RMSNorm<br/>(feeds next layer)"]
        F1 --> F2 --> F3 --> F4 --> F5 --> F6 --> F7
    end

    subgraph SAVINGS["DRAM Traffic Savings"]
        S1["15 kernel launches → 7"]
        S2["Eliminated intermediates:<br/>norm_out, S matrix, P matrix,<br/>residual sum (×2)"]
        S3["At M=512, D=4096:<br/>~100 MiB saved per layer"]
        S4["32 layers × 100 MiB<br/>= 3.2 GiB less DRAM traffic<br/>per forward pass"]
    end

    UNFUSED -.->|"optimizes to"| FUSED
    FUSED -.->|"quantified"| SAVINGS

    style UNFUSED fill:#4a1a1a,stroke:#6a3a3a,color:#e0e0e0
    style FUSED fill:#1a4a1a,stroke:#3a6a3a,color:#e0e0e0
    style SAVINGS fill:#1a1a4a,stroke:#3a3a6a,color:#e0e0e0
```

---

## 9. Quantization — From FP16 to INT4

```mermaid
flowchart TB
    subgraph FP_WORLD["Full Precision World"]
        FP0["FP32 Weights<br/>(training output)"]
        FP1["FP16/BF16 Weights<br/>(direct cast, minimal loss)"]
        FP2["Model size: ~14 GiB<br/>(LLaMA-7B in FP16)"]
    end

    subgraph QUANT_PROCESS["Quantization Process"]
        Q0["Choose block size<br/>(typically 32 or 128 elements)"]
        Q1["For each block:<br/>Find min and max"]
        Q2["Compute scale = (max - min) / (2^bits - 1)<br/>Compute zero_point = round(-min / scale)"]
        Q3["Quantize: q = round(w / scale) + zero_point<br/>Clamp to valid range"]
        Q4["Pack: 2 INT4 values per byte"]
        Q0 --> Q1 --> Q2 --> Q3 --> Q4
    end

    subgraph DEQUANT["Dequantization at Runtime"]
        D0["Unpack nibbles<br/>(shift and mask operations)"]
        D1["Dequant: w = (q - zero_point) × scale"]
        D2["Result: FP16/FP32 for GEMM"]
        D3["Fused dequant + GEMM kernel:<br/>dequantize inside the GEMM inner loop<br/>to avoid materializing FP16 weights"]
        D0 --> D1 --> D2
        D1 --> D3
    end

    subgraph FORMATS["Common Quantization Formats"]
        QF0["Q4_0: 4-bit, block=32,<br/>1 FP16 scale per block"]
        QF1["Q4_K_M: 4-bit, block=256,<br/>6-bit super-scales,<br/>4-bit sub-scales + mins"]
        QF2["Q8_0: 8-bit, block=32,<br/>1 FP16 scale per block"]
        QF3["GPTQ: 4-bit, column-wise,<br/>group_size=128, learned scales"]
        QF4["AWQ: 4-bit, activation-aware,<br/>protects salient weight channels"]
    end

    subgraph SIZES["Memory Footprint (LLaMA-7B)"]
        S0["FP32: 28 GiB"]
        S1["FP16: 14 GiB"]
        S2["INT8: 7 GiB"]
        S3["INT4: 3.8 GiB<br/>(fits in 4GB VRAM!)"]
    end

    FP0 --> FP1
    FP1 --> QUANT_PROCESS
    QUANT_PROCESS --> DEQUANT
    QUANT_PROCESS -.-> FORMATS
    FORMATS -.-> SIZES

    style FP_WORLD fill:#3a3a1a,stroke:#5a5a2a,color:#e0e0e0
    style QUANT_PROCESS fill:#1a3a3a,stroke:#2a5a5a,color:#e0e0e0
    style DEQUANT fill:#3a1a3a,stroke:#5a2a5a,color:#e0e0e0
    style FORMATS fill:#2a3a2a,stroke:#4a5a4a,color:#e0e0e0
    style SIZES fill:#3a2a2a,stroke:#5a4a4a,color:#e0e0e0
```

---

## 10. Full Autoregressive Decode Loop — Timing Breakdown

```mermaid
flowchart TB
    START["Prefill complete.<br/>Have hidden state at position S-1."]

    subgraph DECODE_LOOP["Decode Loop (repeat until EOS)"]
        DL0["Embed token t<br/>(~0 FLOP, table lookup)"]

        subgraph PER_LAYER["× 32 Layers"]
            L0["Residual RMSNorm<br/>(memory-bound, ~1 µs)"]
            L1["QKV Projection<br/>(1 × 4096 × 12288 GEMM,<br/>~100 GFLOP, memory-bound at M=1)"]
            L2["RoPE<br/>(elementwise, ~0.5 µs)"]
            L3["KV Cache Append<br/>(2 × head_dim writes per head)"]
            L4["Attention Score<br/>(1 × t dot products per head,<br/>grows linearly with context)"]
            L5["Softmax<br/>(over t scores)"]
            L6["Value Weighting<br/>(1 × head_dim per head)"]
            L7["Output Projection<br/>(GEMM: 1 × 4096 × 4096)"]
            L8["Residual Add"]
            L9["RMSNorm"]
            L10["SwiGLU FFN<br/>(3 GEMMs: gate, up, down<br/>1 × 4096 → 11008 → 4096)"]
            L11["Residual Add"]
            L0 --> L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8 --> L9 --> L10 --> L11
        end

        DL1["Final RMSNorm<br/>(~1 µs)"]
        DL2["LM Head GEMM<br/>(1 × 4096 × 32000,<br/>vocab projection)"]
        DL3["Sample Token<br/>(CPU: argmax or multinomial)"]
        DL4{"token == EOS<br/>or pos >= max?"}

        DL0 --> PER_LAYER --> DL1 --> DL2 --> DL3 --> DL4
        DL4 -- "No: continue" --> DL0
    end

    DL4 -- "Yes: done" --> DONE["Output generated sequence"]
    START --> DECODE_LOOP

    subgraph BOTTLENECK["Where Time Goes (M=1 Decode)"]
        BN0["~90% Weight loading from DRAM<br/>(memory-bandwidth bound)"]
        BN1["~5% KV cache reads<br/>(grows with sequence length)"]
        BN2["~3% Kernel launch overhead<br/>(~5 µs × ~200 launches)"]
        BN3["~2% Actual arithmetic<br/>(ALUs are starved)"]
    end

    DECODE_LOOP -.-> BOTTLENECK

    style DECODE_LOOP fill:#1a2a1a,stroke:#2a4a2a,color:#e0e0e0
    style PER_LAYER fill:#2a3a2a,stroke:#4a5a4a,color:#e0e0e0
    style BOTTLENECK fill:#4a2a1a,stroke:#6a4a2a,color:#e0e0e0
```

---

## 11. Performance Bound Analysis — Roofline Model

```mermaid
flowchart LR
    subgraph COMPUTE["Compute-Bound Operations"]
        CB0["Prefill GEMMs (large M)<br/>AI >> balance point"]
        CB1["matmul_tiled: 2563 GFLOP/s<br/>vs cuBLAS: 24117 GFLOP/s"]
        CB2["Bottleneck: FP32 throughput,<br/>register pressure,<br/>shared memory reuse"]
    end

    subgraph MEMORY["Memory-Bound Operations"]
        MB0["Decode GEMMs (M=1)<br/>AI < balance point"]
        MB1["RMSNorm: 435 GB/s<br/>(86% of peak BW)"]
        MB2["Softmax: 435 GB/s<br/>(86% of peak BW)"]
        MB3["Bottleneck: DRAM bandwidth,<br/>cannot compute faster than<br/>we can feed data"]
    end

    subgraph BALANCE["RTX 4070 SUPER Balance Point"]
        BP0["Peak compute: 82.6 TFLOP/s FP32"]
        BP1["Peak bandwidth: 504 GB/s"]
        BP2["Balance: 82600 / 504 = 164 FLOP/byte"]
        BP3["If AI < 164: memory-bound<br/>If AI > 164: compute-bound"]
    end

    subgraph PRACTICAL["Practical Implications"]
        PR0["Single-token decode is<br/>ALWAYS memory-bound<br/>(AI ~ 1-2 FLOP/byte)"]
        PR1["Prefill can be compute-bound<br/>(AI ~ 170+ FLOP/byte for large M)"]
        PR2["Optimization priority:<br/>1. Reduce memory traffic (fusion)<br/>2. Reduce launches (fusion)<br/>3. Increase arithmetic efficiency"]
    end

    COMPUTE ~~~ MEMORY
    BALANCE --> PRACTICAL

    style COMPUTE fill:#1a4a1a,stroke:#2a6a2a,color:#e0e0e0
    style MEMORY fill:#4a1a1a,stroke:#6a2a2a,color:#e0e0e0
    style BALANCE fill:#1a1a4a,stroke:#2a2a6a,color:#e0e0e0
    style PRACTICAL fill:#3a3a1a,stroke:#5a5a2a,color:#e0e0e0
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| Solid arrow (`-->`) | Data flow / execution order |
| Dashed arrow (`-.->`) | Conceptual relationship / annotation |
| Rectangular box | Computation step |
| Diamond `{ }` | Decision / branch |
| Subgraph | Logical grouping of related steps |
| `⊙` | Elementwise (Hadamard) product |
| `×` | Matrix multiplication |
| `σ(x)` | Sigmoid function: `1 / (1 + exp(-x))` |
| `SiLU(x)` | `x × σ(x)` — Sigmoid Linear Unit |
| `AI` | Arithmetic Intensity (FLOP / byte) |

---

> **Note:** This flowchart describes the LLaMA / Llama-2 / Mistral family architecture
> specifically, as that is the target of the bare-metal-llm-engine project. Other
> architectures (GPT-2, Falcon, MPT) differ in details (LayerNorm vs RMSNorm, MHA vs GQA,
> GELU vs SiLU, parallel vs sequential attention+FFN) but share the same high-level
> structure of embedding → N × transformer layer → logits → sample → loop.

---

## 12. Streaming Detokenization & UTF-8 Reassembly Buffer

In real-world inference systems, tokens are streamed to the client immediately as they are generated rather than waiting for EOS. Because BPE vocabularies partition arbitrary byte sequences, multi-byte Unicode characters (e.g., emojis or CJK characters spanning 2–4 bytes) are frequently split across multiple consecutive tokens. Emitting decoded text without reassembly causes unicode decoding errors or corrupt replacement characters (`\uFFFD`).

```mermaid
flowchart TB
    subgraph DECODE_STEP["Inside Autoregressive Decode Loop (Token t)"]
        T_NEW["Sampled Token ID: token_t"]
        T_EMIT{"Is token_t == EOS?"}
        T_NEW --> T_EMIT
    end

    subgraph UTF8_STATE["UTF-8 State Machine & Reassembly Buffer"]
        direction TB
        B_LOOKUP["Lookup Raw Bytes in Tokenizer Table<br/>(bytes_t ∈ Vocab)"]
        B_APPEND["Append bytes_t to accumulator buffer:<br/>buffer += bytes_t"]
        B_VALIDATE{"Inspect buffer bytes:<br/>Forms complete, valid<br/>UTF-8 code points?"}
        
        B_HOLD["Hold bytes in buffer<br/>(Incomplete multi-byte sequence,<br/>e.g., byte 1 of 4 in emoji)"]
        B_DECODE["Decode valid UTF-8 slice to string<br/>(e.g., text_chunk)"]
        B_TRIM["Shift buffer:<br/>Remove decoded bytes, keep remainder"]
        
        B_LOOKUP --> B_APPEND --> B_VALIDATE
        B_VALIDATE -- "No (incomplete)" --> B_HOLD
        B_VALIDATE -- "Yes (valid)" --> B_DECODE --> B_TRIM
    end

    subgraph CLIENT_OUTPUT["Client Stream Interface"]
        STREAM_YIELD["Yield text_chunk to client / stdout<br/>(Zero latency, immediate display)"]
        STREAM_CLOSE["Flush any remaining buffer & Close Stream"]
    end

    T_EMIT -- "No (continue)" --> B_LOOKUP
    T_EMIT -- "Yes (EOS)" --> STREAM_CLOSE
    B_TRIM --> STREAM_YIELD
    B_HOLD -.->|"Wait for next token"| DECODE_STEP

    style DECODE_STEP fill:#1a2a3a,stroke:#2a4a6a,color:#e0e0e0
    style UTF8_STATE fill:#2a1a3a,stroke:#5a2a6a,color:#e0e0e0
    style CLIENT_OUTPUT fill:#1a3a2a,stroke:#2a5a4a,color:#e0e0e0
```

---

## 13. CUDA Graphs & Runtime Orchestration (Zero-Overhead Decode)

In single-token decode ($M=1$), the GPU executes kernels in fractions of a microsecond. Issuing ~15–20 kernels per layer across 32 layers requires ~500–600 individual CUDA kernel launches per token. CPU driver launch latency (~3–5 µs per launch) starves the GPU. Industry engines eliminate this bottleneck by baking the entire decode loop into a pre-compiled **CUDA Graph**.

```mermaid
flowchart TB
    subgraph WARMUP["1. Capture Phase (Run Once at Startup)"]
        W0["Allocate static device memory for M=1 decode:<br/>d_token_id, d_logits, d_kv_cache, d_scratch"]
        W1["cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)"]
        W2["Execute complete forward pass (all 32 layers):<br/>- RMSNorm, QKV GEMV, RoPE<br/>- KV append, Attention, Output GEMV<br/>- SwiGLU FFN (Gate/Up/Down GEMV)"]
        W3["cudaStreamEndCapture(stream, &graph)"]
        W4["cudaGraphInstantiate(&graphExec, graph, 0)<br/>(Bakes topology directly into GPU command processor list)"]
        W0 --> W1 --> W2 --> W3 --> W4
    end

    subgraph EXEC["2. Execution Phase (Autoregressive Loop per Token)"]
        E0["Host updates input token ID in pinned memory<br/>or async H2D copy to static d_token_id"]
        E1["Update dynamic graph node parameters if needed<br/>(e.g., current sequence position slot t)"]
        E2["cudaGraphLaunch(graphExec, stream)<br/>(Single IOCTL / kernel dispatch call for all 32 layers)"]
        E3["Logits ready on device: d_logits<br/>Launch sampling kernel directly on stream"]
        E0 --> E1 --> E2 --> E3
    end

    subgraph COMPARISON["Dispatch Latency Comparison"]
        direction LR
        subgraph STANDARD["Standard Sequential Decode Step"]
            S_CPU["CPU launches ~200 kernels sequentially"]
            S_TIME["CPU overhead: ~500–1000 µs<br/>(CPU throttles GPU execution)"]
            S_CPU --> S_TIME
        end
        subgraph GRAPHED["CUDA Graph Decode Step"]
            G_CPU["CPU launches 1 single graph"]
            G_TIME["CPU overhead: ~10–15 µs<br/>(GPU fully saturated at 100% memory BW)"]
            G_CPU --> G_TIME
        end
    end

    W4 --> E0
    E3 -.-> COMPARISON

    style WARMUP fill:#1a3a3a,stroke:#2a5a5a,color:#e0e0e0
    style EXEC fill:#2a3a1a,stroke:#4a5a2a,color:#e0e0e0
    style STANDARD fill:#4a1a1a,stroke:#6a2a2a,color:#e0e0e0
    style GRAPHED fill:#1a4a1a,stroke:#2a6a2a,color:#e0e0e0
```

---

## 14. FlashAttention & Online Softmax Algorithm (SRAM Tiling)

Standard self-attention materializes the $S \times S$ attention matrix in GPU DRAM, requiring $O(S^2)$ memory reads and writes. FlashAttention solves this via **tiling in Shared Memory (SRAM)** and the **Online Softmax** algorithm, which updates running row statistics on-the-fly without ever writing intermediate scores to global DRAM.

```mermaid
flowchart TB
    subgraph DRAM_INPUT["Global Memory (DRAM / HBM)"]
        D_Q["Q Matrix (seq_q × head_dim)"]
        D_K["K Matrix (seq_k × head_dim)"]
        D_V["V Matrix (seq_k × head_dim)"]
        D_O["Output O (seq_q × head_dim)"]
    end

    subgraph SRAM_LOOP["Streaming Multiprocessor SRAM (Shared Memory & Registers)"]
        direction TB
        L0["Divide Q into blocks Q_i of size B_r × d<br/>Divide K, V into blocks K_j, V_j of size B_c × d"]
        
        subgraph OUTER["Outer Loop: Load Q_i into SRAM"]
            O_INIT["Initialize running row statistics in registers:<br/>m_i = -∞ (running row max)<br/>l_i = 0  (running row sum)<br/>O_i = 0  (running unnormalized output accumulator)"]
            
            subgraph INNER["Inner Loop: Iterate over K_j, V_j blocks"]
                I_LOAD["Load K_j, V_j from DRAM into Shared Memory"]
                I_GEMM["Compute local tile scores:<br/>S_ij = Q_i × K_j^T / sqrt(d)  (B_r × B_c)"]
                I_MASK["Apply causal mask to S_ij if row index < col index"]
                I_MAX["Find new row max: m_new = max(m_i, rowmax(S_ij))"]
                I_RESCALE["Rescale previous accumulator:<br/>P_scale = exp(m_i - m_new)"]
                I_EXP["Compute local probabilities:<br/>P_ij = exp(S_ij - m_new)"]
                I_SUM["Update running sum:<br/>l_new = P_scale × l_i + rowsum(P_ij)"]
                I_ACC["Update output accumulator:<br/>O_i = diag(P_scale) × O_i + P_ij × V_j"]
                I_UPDATE["Update running state: m_i = m_new, l_i = l_new"]
                
                I_LOAD --> I_GEMM --> I_MASK --> I_MAX --> I_RESCALE --> I_EXP --> I_SUM --> I_ACC --> I_UPDATE
            end
            
            O_NORM["Final row normalization in registers:<br/>O_i = diag(1 / l_i) × O_i"]
            O_WRITE["Write O_i back to DRAM (Single global write!)"]
            
            O_INIT --> INNER --> O_NORM --> O_WRITE
        end
    end

    D_Q --> L0
    D_K --> L0
    D_V --> L0
    L0 --> OUTER
    O_WRITE --> D_O

    subgraph BENEFIT["Why Online Softmax Is Revolutionary"]
        B1["Standard Attention: Materializes S (N × N) and P (N × N) in DRAM<br/>Memory Complexity: O(N²), DRAM Traffic: O(N²)"]
        B2["FlashAttention Online Softmax: Never writes S or P to DRAM<br/>Memory Complexity: O(N) auxiliary, DRAM Traffic: O(N² / SRAM_size)"]
    end

    OUTER -.-> BENEFIT

    style DRAM_INPUT fill:#2a2a3a,stroke:#4a4a5a,color:#e0e0e0
    style SRAM_LOOP fill:#1e2a4a,stroke:#3a5a8a,color:#e0e0e0
    style INNER fill:#2a3a5a,stroke:#4a6a9a,color:#e0e0e0
    style BENEFIT fill:#3a2a1a,stroke:#5a4a2a,color:#e0e0e0
```

---

## 15. FlashDecoding for Long-Context Decode (M=1 Parallelism)

FlashAttention parallelizes across batch size and query attention heads. In the prefill phase ($M=S$), there is abundant query parallelism. In the decode phase ($M=1$), a batch of 1 with 32 heads provides only 32 thread blocks—leaving modern GPUs with 56 SMs (RTX 4070S) or 132 SMs (H100) severely underutilized. **FlashDecoding** solves this by parallelizing across the KV sequence length dimension.

```mermaid
flowchart TB
    subgraph PROBLEM["The Decode Bottleneck for Standard FlashAttention"]
        P0["In Decode: Query length M = 1<br/>Batch = 1, Heads = 32"]
        P1["Total thread blocks in standard FlashAttention = Batch × Heads = 32 blocks"]
        P2["GPU has 56 SMs (RTX 4070S) or 132 SMs (H100)"]
        P3["Result: 24+ SMs sit completely IDLE! GPU compute is starved."]
        P0 --> P1 --> P2 --> P3
    end

    subgraph FLASH_DECODE["FlashDecoding: Parallelize Along KV Sequence Dimension"]
        direction TB
        FD0["Split KV Cache (length S) into K equal chunks of size B_c (e.g., 256)"]
        
        subgraph STAGE1["Stage 1: Parallel Partial Attention (All SMs Fully Saturated)"]
            direction LR
            CB0["Thread Block 0<br/>Head h, KV chunk 0<br/>Computes (O_0, l_0, m_0)"]
            CB1["Thread Block 1<br/>Head h, KV chunk 1<br/>Computes (O_1, l_1, m_1)"]
            CB2["Thread Block K-1<br/>Head h, KV chunk K-1<br/>Computes (O_K-1, l_K-1, m_K-1)"]
        end
        
        FD1["Grid size = Batch × Heads × K chunks<br/>Example: 1 × 32 × 16 chunks = 512 blocks!<br/>All 56 SMs saturated at high occupancy."]
        
        subgraph STAGE2["Stage 2: Reduction Kernel (Merge Partial Outputs)"]
            R0["Read K partial results (O_k, l_k, m_k) from scratch DRAM buffer"]
            R1["Global max: m_global = max_k(m_k)"]
            R2["Rescale sums: l_global = sum_k(exp(m_k - m_global) × l_k)"]
            R3["Weighted output: O_final = sum_k(exp(m_k - m_global) × O_k) / l_global"]
            R0 --> R1 --> R2 --> R3
        end
        
        FD0 --> STAGE1 --> FD1 --> STAGE2
    end

    PROBLEM -.->|"Solved by"| FLASH_DECODE

    style PROBLEM fill:#4a1a1a,stroke:#6a2a2a,color:#e0e0e0
    style FLASH_DECODE fill:#1a3a2a,stroke:#2a5a4a,color:#e0e0e0
    style STAGE1 fill:#2a4a3a,stroke:#3a6a5a,color:#e0e0e0
    style STAGE2 fill:#1a2a4a,stroke:#2a4a6a,color:#e0e0e0
```

---

## 16. PagedAttention & Dynamic KV Cache Memory Management

Static contiguous KV cache allocation pre-reserves memory for `max_seq_len` tokens per sequence. In production, request lengths vary wildly, leading to 60–80% VRAM waste due to internal and external fragmentation. Inspired by OS virtual memory paging, **PagedAttention** partitions the KV cache into fixed-size physical blocks allocated dynamically.

```mermaid
flowchart TB
    subgraph CONTIGUOUS_VS_PAGED["Memory Allocation Comparison"]
        direction LR
        subgraph CONTIG["Contiguous Static Buffer (Naive)"]
            C0["Pre-allocate max_seq_len (e.g., 4096 tokens)"]
            C1["Prompt: 100 tokens. Active decode: 50 tokens.<br/>Wasted slots: 3946 tokens (96% VRAM wasted!)"]
            C2["Cannot share system prompt prefixes across requests"]
            C0 --> C1 --> C2
        end
        subgraph PAGED["PagedAttention (Virtual Memory for KV Cache)"]
            PG0["Allocate physical memory in fixed-size blocks (block_size = 16 tokens)"]
            PG1["Allocate blocks strictly on demand as new tokens are generated"]
            PG2["Memory fragmentation drops to < 4%"]
            PG0 --> PG1 --> PG2
        end
    end

    subgraph ARCHITECTURE["PagedAttention Architecture & Data Structures"]
        direction TB
        
        subgraph LOGICAL["Logical KV Cache (Per Sequence / Request)"]
            L_B0["Logical Block 0 (Tokens 0..15)"]
            L_B1["Logical Block 1 (Tokens 16..31)"]
            L_B2["Logical Block 2 (Tokens 32..47)"]
        end

        subgraph BLOCK_TABLE["Block Table (Page Table per Request)"]
            BT["Logical Block 0 ➔ Physical Block 7<br/>Logical Block 1 ➔ Physical Block 2<br/>Logical Block 2 ➔ Physical Block 13"]
        end

        subgraph PHYSICAL_POOL["Physical Block Pool (Pre-allocated VRAM Pool)"]
            P_B2["Physical Block 2 (16 tokens K,V)"]
            P_B7["Physical Block 7 (16 tokens K,V)"]
            P_B13["Physical Block 13 (16 tokens K,V)"]
            P_FREE["Free Blocks Queue: [0, 1, 3, 4, 5, 6, 8, 9, 10, ...]"]
        end

        L_B0 --> BT
        L_B1 --> BT
        L_B2 --> BT
        BT --> P_B7
        BT --> P_B2
        BT --> P_B13
    end

    subgraph KERNEL_FETCH["Paged Attention Kernel Execution"]
        K0["Thread determines current token index t"]
        K1["logical_block = t / 16,  offset = t % 16"]
        K2["physical_block = block_table[request_id][logical_block]"]
        K3["Fetch K,V from physical_pool[physical_block][offset]"]
        K0 --> K1 --> K2 --> K3
    end

    CONTIGUOUS_VS_PAGED --> ARCHITECTURE --> KERNEL_FETCH

    style CONTIG fill:#4a2a1a,stroke:#6a3a2a,color:#e0e0e0
    style PAGED fill:#1a4a2a,stroke:#2a6a3a,color:#e0e0e0
    style ARCHITECTURE fill:#1a2a4a,stroke:#2a4a6a,color:#e0e0e0
    style KERNEL_FETCH fill:#2a1a4a,stroke:#4a2a6a,color:#e0e0e0
```

---

## 17. KV Cache Quantization (FP8 & INT8)

As context windows extend to 32k–128k tokens, the memory occupied by the KV cache surpasses the model weights themselves. Quantizing the KV cache to 8 bits (FP8 or INT8) cuts memory traffic and footprint in half, allowing models to support double the context length or double the concurrent batch size.

```mermaid
flowchart TB
    subgraph MEM_CRISIS["The Long-Context VRAM Wall (FP16 KV Cache)"]
        M0["Model: LLaMA-3-8B (32 layers, 8 KV heads, 128 head_dim)"]
        M1["FP16 KV Cache at 4K context: ~1.0 GiB"]
        M2["FP16 KV Cache at 32K context: ~8.0 GiB (Exceeds weights of INT4 model!)"]
        M3["FP16 KV Cache at 128K context: ~32.0 GiB (Cannot fit in consumer VRAM!)"]
        M0 --> M1 --> M2 --> M3
    end

    subgraph FORMATS_KV["KV Quantization Numeric Dtypes"]
        direction LR
        subgraph FP8_E4M3["FP8 (E4M3) — Modern Standard (Ada / Hopper)"]
            F0["1 sign bit, 4 exponent bits, 3 mantissa bits"]
            F1["Dynamic range: [-448, 448]"]
            F2["Direct hardware Tensor Core support on RTX 40 & H100"]
            F3["50% memory reduction with negligible perplexity loss"]
            F0 --> F1 --> F2 --> F3
        end
        subgraph INT8_PER_TENSOR["INT8 Per-Token / Per-Channel"]
            I0["1 sign bit, 7 magnitude bits"]
            I1["scale_k = max(|K|) / 127"]
            I2["Quant: q_k = round(K / scale_k)"]
            I3["Requires explicit scaling multiply inside dot product"]
            I0 --> I1 --> I2 --> I3
        end
    end

    subgraph FUSED_ATTN_QUANT["Fused Quantized KV Attention Kernel"]
        Q0["Query Q kept in FP16/BF16"]
        Q1["Fetch packed FP8/INT8 K, V from cache memory (half the DRAM bytes!)"]
        Q2["Dequantize on-the-fly in registers / shared memory"]
        Q3["Compute Q × K^T dot products with native FP8 or FP16 math"]
        Q0 --> Q2
        Q1 --> Q2 --> Q3
    end

    MEM_CRISIS --> FORMATS_KV --> FUSED_ATTN_QUANT

    style MEM_CRISIS fill:#4a1a2a,stroke:#6a2a3a,color:#e0e0e0
    style FP8_E4M3 fill:#1a3a4a,stroke:#2a5a6a,color:#e0e0e0
    style INT8_PER_TENSOR fill:#3a2a4a,stroke:#5a3a6a,color:#e0e0e0
    style FUSED_ATTN_QUANT fill:#2a4a1a,stroke:#4a6a2a,color:#e0e0e0
```

---

## 18. Speculative Decoding Architecture

Because single-token autoregressive decode is strictly memory-bandwidth bound (reading all ~14 GB of weights for every single token), it cannot saturate modern compute cores. **Speculative Decoding** solves this by pairing a fast draft model with the target model, using the target model to verify $K$ tokens in a single parallel, compute-bound forward pass.

```mermaid
flowchart TB
    subgraph MOTIVATION["The Speculative Opportunity"]
        MOT0["Target Model Decode is Memory-Bound: Generates 1 token per full weight load (AI ~ 1)"]
        MOT1["Idea: Fast draft model speculates K tokens, then target model verifies ALL K tokens in PARALLEL<br/>in a single forward pass (Matrix-Matrix GEMM, AI ~ K!)"]
        MOT0 --> MOT1
    end

    subgraph SPEC_LOOP["Speculative Decoding Iteration (Draft ➔ Verify ➔ Accept)"]
        direction TB
        
        subgraph DRAFT_PHASE["Step 1: Draft Phase (K = 4 tokens)"]
            D0["Small Draft Model (e.g. 68M params or speculative heads)"]
            D1["Run K fast autoregressive steps (~5x faster per token)"]
            D2["Emit candidate sequence: [x_1, x_2, x_3, x_4]"]
            D0 --> D1 --> D2
        end

        subgraph TARGET_VERIFY["Step 2: Target Model Parallel Verification"]
            T0["Input full sequence [x_0, x_1, x_2, x_3, x_4] to Target Model (7B/8B)"]
            T1["Run SINGLE forward pass with M = 5 (Compute-bound GEMM!)"]
            T2["Obtain target model probability distributions:<br/>[p_0(x), p_1(x), p_2(x), p_3(x), p_4(x)]"]
            T0 --> T1 --> T2
        end

        subgraph ACCEPT_LOGIC["Step 3: Rejection Sampling / Acceptance Engine"]
            A0["For i = 1 to K:<br/>Compare draft prob q_i(x_i) vs target prob p_i(x_i)"]
            A1{"Uniform(0,1) < min(1, p_i / q_i)?"}
            A2["Accept x_i, advance position"]
            A3["Reject x_i:<br/>Sample next token from max(0, p_i - q_i)<br/>Discard remaining draft tokens"]
            A4["Update KV cache: Keep accepted tokens, discard rejected slots"]
            
            A0 --> A1
            A1 -- "Yes" --> A2 --> A0
            A1 -- "No" --> A3 --> A4
        end

        D2 --> TARGET_VERIFY --> ACCEPT_LOGIC
    end

    subgraph SPEEDUP["Performance Outcome"]
        SP0["Average Acceptance Rate: α ≈ 70–80%"]
        SP1["Tokens generated per target model step: 1 + α × K ≈ 3.2 tokens"]
        SP2["Effective Wall-Clock Latency: 2.0x – 2.8x faster generation"]
        SP0 --> SP1 --> SP2
    end

    MOTIVATION --> SPEC_LOOP --> SPEEDUP

    style MOTIVATION fill:#2a2a4a,stroke:#4a4a6a,color:#e0e0e0
    style DRAFT_PHASE fill:#3a2a1a,stroke:#5a4a2a,color:#e0e0e0
    style TARGET_VERIFY fill:#1a3a3a,stroke:#2a5a5a,color:#e0e0e0
    style ACCEPT_LOGIC fill:#3a1a3a,stroke:#5a2a5a,color:#e0e0e0
    style SPEEDUP fill:#1a4a1a,stroke:#2a6a2a,color:#e0e0e0
```

---

## 19. Continuous Batching & Chunked Prefill (Serving Architecture)

When scaling inference beyond a single user to a multi-tenant serving system, traditional static batching introduces massive idle time because requests have heterogeneous lengths. Industry engines use **Continuous (Iteration-Level) Batching** and **Chunked Prefill** to maximize throughput while maintaining tight latency guarantees.

```mermaid
flowchart TB
    subgraph BATCHING_TYPES["Static Batching vs Continuous Batching"]
        direction LR
        subgraph STATIC["Static Batching (Traditional)"]
            ST0["Wait for B requests to accumulate"]
            ST1["Pad all prompts to longest length in batch"]
            ST2["All requests must wait until the SLOWEST request finishes generating"]
            ST3["Severe GPU idle time during tail generation"]
            ST0 --> ST1 --> ST2 --> ST3
        end
        subgraph CONTINUOUS["Continuous / Iteration-Level Batching (Orca / vLLM)"]
            CB0["Scheduler operates at individual iteration (token) granularity"]
            CB1["Finished requests immediately exit and free KV memory"]
            CB2["New requests immediately join the batch on the next decode step"]
            CB3["No padding tokens needed"]
            CB0 --> CB1 --> CB2 --> CB3
        end
    end

    subgraph CHUNKED_PREFILL["Chunked Prefill & Piggybacking (Sarathi)"]
        direction TB
        CP0["Problem: Prefill is compute-heavy (high latency spike);<br/>Decode is latency-sensitive (requires smooth Inter-Token Latency ITL)"]
        CP1["Solution: Chunk long prefill into slices of size C (e.g. 512 tokens)"]
        CP2["Batch composition per step:<br/>Chunked Prefill (512 tokens) + Active Decodes (N tokens)"]
        CP3["Maintains optimal GPU Arithmetic Intensity while eliminating ITL latency bubbles"]
        CP0 --> CP1 --> CP2 --> CP3
    end

    BATCHING_TYPES --> CHUNKED_PREFILL

    style STATIC fill:#4a1a1a,stroke:#6a2a2a,color:#e0e0e0
    style CONTINUOUS fill:#1a4a1a,stroke:#2a6a2a,color:#e0e0e0
    style CHUNKED_PREFILL fill:#1a2a4a,stroke:#2a4a6a,color:#e0e0e0
```

---

## 20. Implementation Complexity Spectrum — Bare Metal vs Industry Serving

| Component | Educational / Minimal Bare-Metal Engine | Production Serving Engine (vLLM / TensorRT-LLM) | Why Production Added It |
|---|---|---|---|
| **KV Cache Layout** | Contiguous static allocation (`max_seq_len`) | PagedAttention (virtual page tables, blocks of 16 tokens) | Eliminates 60–80% VRAM memory waste & fragmentation |
| **Kernel Dispatch** | Host loop calling individual CUDA kernels | CUDA Graphs (`cudaGraphLaunch`) | Drops CPU dispatch overhead from ~500 µs to ~10 µs per token |
| **Attention Kernel** | Standard GEMM or basic FlashAttention | FlashDecoding + Split-KV reduction | Prevents SM underutilization during single-token decode ($M=1$) |
| **Detokenization** | Decode complete array once at EOS | Streaming multi-byte UTF-8 accumulator state machine | Immediate token streaming without emitting corrupted bytes (`\uFFFD`) |
| **Batching Strategy** | Single-stream sequential request | Iteration-level Continuous Batching + Chunked Prefill | Eliminates padding waste and prevents prefill head-of-line blocking |
| **Generation Algorithm** | Strict 1-token-at-a-time autoregression | Speculative Decoding (Draft model + Target verify) | Breaks memory-bandwidth bottleneck to achieve 2–3× token/sec |
| **KV Cache Dtype** | FP16 / BF16 (2 bytes / element) | FP8 E4M3 / INT8 per-channel quantization | Doubles or quadruples effective context window in same VRAM |

