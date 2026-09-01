#!/usr/bin/env bash
#
# The Machine B (RTX 4090) run script: pull -> configure -> build -> generate
# reference data -> test -> benchmark, with the results logged.
#
#   scripts/gpu-run.sh                  # the whole sequence
#   scripts/gpu-run.sh --no-pull        # benchmark what is already checked out
#   scripts/gpu-run.sh --no-bench       # correctness only
#   scripts/gpu-run.sh --allow-dirty    # override the clean-tree requirement
#
# Under WSL2 or Git Bash on Windows. For a native Windows shell use
# scripts/build.ps1, which handles the MSVC environment; this script assumes a
# working POSIX shell and that cmake/nvcc are already on PATH.
#
# THE FOUR RULES THIS SCRIPT ENFORCES
#
#   1. Benchmark committed source, never a dirty tree. Six months from now the lab
#      notebook will say "0.9 ms at commit a1b2c3d". If the tree was dirty, that
#      sentence is false and the measurement is unreproducible. Git is the transport
#      between the two machines (docs/01-dev-environment.md); local edits on the GPU
#      box quietly break that.
#
#   2. Correctness before speed, enforced by ordering. The benchmark does not run if
#      the tests fail. Optimising an incorrect kernel is the most expensive mistake
#      available here: you tune the wrong arithmetic and then have to redo it.
#
#   3. Record the machine state next to the numbers. Device, driver, clocks and
#      persistence mode go into the log above the tables, because two runs at
#      different clocks are not comparable and you will not remember which was which.
#
#   4. Run the CPU baseline on THIS machine. A speedup figure that divides 4090
#      kernel times by the laptop's CPU times is not a speedup, it is two unrelated
#      measurements. bench_cpu_ref runs here so the ratio has one machine in it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="build"
BUILD_TYPE="RelWithDebInfo"
DO_PULL=1
DO_BENCH=1
DO_CPU_BENCH=1
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir|-B)      BUILD_DIR="$2";  shift 2 ;;
    --type|-t)     BUILD_TYPE="$2"; shift 2 ;;
    --no-pull)     DO_PULL=0;       shift ;;
    --no-bench)    DO_BENCH=0;      shift ;;
    --no-cpu-bench) DO_CPU_BENCH=0; shift ;;
    --allow-dirty) ALLOW_DIRTY=1;   shift ;;
    -h|--help)     sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
    *) echo "gpu-run.sh: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done

#-----------------------------------------------------------------------------------
# Rule 1 -- clean tree, then fast-forward
#-----------------------------------------------------------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "gpu-run.sh: not inside a git repository." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  if [[ $ALLOW_DIRTY -eq 0 ]]; then
    echo "gpu-run.sh: the working tree has uncommitted changes." >&2
    echo >&2
    git status --short --untracked-files=no >&2
    echo >&2
    echo "Kernels are written on Machine A and measured here, so this box should only" >&2
    echo "ever be a fast-forward of the remote. Commit and push from Machine A, or" >&2
    echo "pass --allow-dirty if you know why you are doing it (and then say so in the" >&2
    echo "notebook entry -- an unreproducible number needs a footnote)." >&2
    exit 1
  fi
  echo "gpu-run.sh: WARNING -- benchmarking a dirty tree, results are not reproducible."
fi

if [[ $DO_PULL -eq 1 ]]; then
  echo "==> git pull --ff-only"
  # --ff-only rather than a plain pull: if this box has diverged from the remote, the
  # right response is to stop and look, not to silently create a merge commit on the
  # machine that is supposed to be a read-only mirror.
  if ! git pull --ff-only; then
    echo "gpu-run.sh: fast-forward failed -- this box has local commits or has" >&2
    echo "            diverged from the remote. Resolve it before measuring." >&2
    exit 1
  fi
fi

COMMIT="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

mkdir -p runs
LOG="runs/$(date +%Y-%m-%d-%H%M)-${COMMIT}.log"

log()  { printf '%s\n' "$*" | tee -a "$LOG"; }
run()  { printf '\n$ %s\n' "$*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

#-----------------------------------------------------------------------------------
# Rule 3 -- provenance header
#-----------------------------------------------------------------------------------
{
  echo "# gpu-run $(date -Iseconds)"
  echo
  echo "commit    : $COMMIT ($BRANCH)"
  echo "subject   : $(git log -1 --pretty=%s)"
  echo "host      : $(hostname)"
  echo "build type: $BUILD_TYPE"
  echo "build dir : $BUILD_DIR"
} > "$LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
  log ""
  log "## GPU state"
  log ""
  nvidia-smi --query-gpu=name,driver_version,persistence_mode,clocks.gr,clocks.max.gr,clocks.mem,temperature.gpu \
             --format=csv 2>&1 | tee -a "$LOG"

  PERSIST="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [[ "$PERSIST" != "Enabled" ]]; then
    log ""
    log "WARNING: persistence mode is '$PERSIST', so the clocks are not locked."
    log "         Back-to-back runs of an unchanged kernel can differ by 10-20% from"
    log "         boost behaviour alone -- larger than most of what you are trying to"
    log "         measure. From an Administrator prompt on Windows:"
    log "             nvidia-smi -pm 1"
    log "             nvidia-smi -lgc <MHz>     # a clock the card can hold, not max boost"
    log "             ...                       # run benchmarks"
    log "             nvidia-smi -rgc"
    log "         Numbers from this run are usable for correctness, and provisional for"
    log "         anything else. The (!) spread column will show you if it mattered."
  fi
else
  log ""
  log "WARNING: nvidia-smi not found. No GPU state recorded for this run."
fi

#-----------------------------------------------------------------------------------
# Build
#-----------------------------------------------------------------------------------
log ""
log "## Build"
if ! run "$REPO_ROOT/scripts/build.sh" --dir "$BUILD_DIR" --type "$BUILD_TYPE"; then
  log ""
  log "BUILD FAILED -- nothing measured. Log: $LOG"
  exit 1
fi

# A CUDA-less build here means the toolchain is not being found, which is the whole
# point of this machine. Do not let it slide past as a green run.
if [[ ! -x "$BUILD_DIR/bin/bench_kernels" && ! -x "$BUILD_DIR/bin/bench_kernels.exe" ]]; then
  log ""
  log "ERROR: bench_kernels was not built, so CUDA was not detected on the machine"
  log "       whose only job is CUDA. Read the configure summary above -- it prints"
  log "       exactly why. On Windows, the usual cause is a shell without cl.exe on"
  log "       PATH: use an x64 Native Tools Command Prompt, or scripts/build.ps1."
  exit 1
fi

#-----------------------------------------------------------------------------------
# Rule 2 -- correctness gates the benchmark
#-----------------------------------------------------------------------------------
log ""
log "## Tests"
if ! run "$REPO_ROOT/scripts/test.sh" --dir "$BUILD_DIR" --no-build; then
  log ""
  log "TESTS FAILED -- skipping the benchmark on purpose."
  log ""
  log "Optimising a kernel that computes the wrong answer is the most expensive"
  log "mistake available here. Fix correctness first. To find which launch faulted:"
  log ""
  log "    scripts/build.sh --dir build-sync --type Debug --sync-check"
  log "    scripts/test.sh  --dir build-sync --no-build -R kernels"
  log ""
  log "Log: $LOG"
  exit 1
fi

#-----------------------------------------------------------------------------------
# Benchmark
#-----------------------------------------------------------------------------------
if [[ $DO_BENCH -eq 0 ]]; then
  log ""
  log "Benchmark skipped (--no-bench). Log: $LOG"
  exit 0
fi

if [[ "$BUILD_TYPE" == "Debug" ]]; then
  log ""
  log "Refusing to benchmark a Debug build -- the numbers would be meaningless."
  log "Re-run without --type Debug. Log: $LOG"
  exit 1
fi

if [[ $DO_CPU_BENCH -eq 1 ]]; then
  log ""
  log "## CPU baseline (same machine -- rule 4)"
  run "$BUILD_DIR/bin/bench_cpu_ref"
fi

log ""
log "## CUDA kernels"
run "$BUILD_DIR/bin/bench_kernels"

log ""
log "Done. Log: $LOG"
log ""
log "Next: paste the tables into docs/lab-notebook.md under this week's entry, with"
log "the locked clock and your prediction-vs-measurement note. Any row marked (!) has"
log "a spread above 10% and is not a number worth quoting -- find out why first."
