# runbook/ — Operational Procedures

> Step-by-step instructions for repeatable tasks. Every procedure here is designed to
> be **copy-paste executable** — an engineer (or AI agent) who has never done this task
> before should be able to follow it successfully on the first attempt.
>
> **The three-minute rule:** If you can't find and understand the first steps within
> three minutes, the runbook is failing.

## Index

### Build & Test
| Runbook | When to use |
|---|---|
| [Build the project](build.md) | First time setup, or when CMake config changes |
| [Run the test suite](test.md) | After any code change, before committing |
| [Regenerate reference data](regenerate-golden.md) | After modifying `gen_reference.py` or adding new test cases |

### CUDA / GPU
| Runbook | When to use |
|---|---|
| [Run CUDA kernels on Machine B](gpu-workflow.md) | Every time you push new kernel code |
| [Run benchmarks with locked clocks](benchmark.md) | When recording performance numbers, exporting JSON, syncing README, or running `llama-bench` |
| [Profile kernels with Nsight Compute](profile-nsight.md) | When analyzing DRAM throughput, memory roofline, occupancy, or bank conflicts |

### Maintenance
| Runbook | When to use |
|---|---|
| [Add a new kernel](add-kernel.md) | When starting a new exercise in the kernel ladder |
| [Add a new ADR](add-adr.md) | When making a significant architectural decision |

## Template for new runbooks

When adding a new runbook, use this template:

```markdown
# [Title]

> **When to use:** [one sentence]
> **Prerequisites:** [what you need before starting]
> **Time estimate:** [how long this takes]

## Steps

1. Step one
   ```bash
   command to run
   ```
2. Step two
   ...

## Verification

How to confirm the procedure succeeded.

## Rollback

How to undo if something goes wrong.

## Troubleshooting

Common problems and their solutions.
```
