# docs/ — Technical Documentation

> Human-readable technical documentation for the Bare-Metal LLM Inference Engine.
> This folder is the **archival library** — it describes the *what*, *how*, and *why*
> of the project's design.

## Index

### Getting Started
- [**01 — Development Environment**](01-dev-environment.md) — two-machine setup,
  build instructions, toolchain requirements, GPU clock-locking recipe

### Exercises & Learning
- [**02 — CUDA Exercise Ladder**](02-cuda-exercises.md) — six kernels in order, each
  introducing one new idea. Read before starting any kernel work.

### Architecture & Deep Dives
- [**LLM Inference Flowchart**](llm-inference-flowchart.md) — comprehensive end-to-end flowchart of bare-metal LLM execution without external ML runtimes
- [**Empirical Roofline Analysis**](roofline.png) — hardware memory bandwidth vs arithmetic intensity ceilings plotted against all kernels

### Performance & Experimental Record
- [**Lab Notebook**](lab-notebook.md) — weekly entries with predictions, measurements, and analysis (primary evidence)
- [**Negative Results & Empirical Boundaries**](negative-results.md) — catalog of optimizations that failed or degraded throughput and their microarchitectural causes

### Architectural Decision Records (ADRs)
Formal records of significant design decisions and their rationale.

| ADR | Title | Key Takeaway |
|---|---|---|
| [ADR-0001](adr/0001-cuda-optional-build.md) | CUDA is optional | The whole project builds without a GPU — Machine A stays usable |
| [ADR-0002](adr/0002-storage-tensor-split.md) | Storage/Tensor split | Views are free; KV-cache slices, mmap'd weights, and reshapes need this |
| [ADR-0003](adr/0003-raw-pointer-kernel-api.md) | Raw pointer kernel API | Kernels take raw pointers, not Tensor — decouples exercises from Module 1 |
| [ADR-0004](adr/0004-cublas-baseline-only.md) | cuBLAS is baseline only | cuBLAS is in one benchmark target, never in the engine |
| [ADR-0005](adr/0005-cuda-arch-explicit.md) | Explicit CUDA arch | `ENGINE_CUDA_ARCH=89` — wrong-arch builds fail loudly |
| [ADR-0006](adr/0006-kernel-fusion-strategy.md) | Kernel fusion strategy | Target transformer sub-layer boundaries (`RMSNorm+Linear`, `Residual+RMSNorm`) to eliminate intermediate DRAM traffic |
| [ADR-0007](adr/0007-gemv-decode-specialization.md) | GEMV decode specialization | Dedicated matrix-vector kernel for $M=1$ autoregressive decode, achieving 94.0% peak BW and beating cuBLAS |
| [ADR-0008](adr/0008-benchmark-harness-provenance.md) | Benchmark harness JSON & provenance | Structured `--json` export, device driver, hardware clock, and git commit hash tracking |
| [ADR-0009](adr/0009-negative-results-reporting.md) | Negative results reporting | Explicitly document optimizations that failed or degraded performance to preserve empirical boundaries |

### API Reference (to be added)

As modules are completed, API documentation will be added here:

- [ ] `docs/api-tensor.md` — Tensor and Storage API reference
- [ ] `docs/api-kernels.md` — Kernel launcher API reference
- [ ] `docs/api-cpu-ref.md` — CPU oracle API reference

### Guides (to be added)

- [ ] `docs/guide-adding-a-kernel.md` — How to add a new kernel to the ladder
- [ ] `docs/guide-benchmarking.md` — How to run and interpret benchmarks
- [ ] `docs/guide-model-loading.md` — How the GGUF loader will work

## Document standards

1. **Markdown only** — for both human and AI readability
2. **Each doc should be self-contained** — a reader shouldn't need to jump to three
   other files to understand one concept
3. **Include code examples** — show, don't just tell
4. **Date your entries** — especially in the lab notebook
5. **Link to source** — reference the actual files in the codebase where applicable

## Relationship to other folders

| Folder | Purpose | Audience |
|---|---|---|
| `docs/` (this) | Technical reference, decisions, learning record | Humans |
| `brain/` | Persistent AI context, project state, conventions | AI agents + humans |
| `runbook/` | Step-by-step operational procedures | Anyone executing tasks |
