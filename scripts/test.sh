#!/usr/bin/env bash
#
# Build, make sure the reference data exists, and run the test suite.
#
#   scripts/test.sh                     # everything
#   scripts/test.sh -R kernels          # just the kernels suite
#   scripts/test.sh --no-build          # skip the build, just test
#   scripts/test.sh --regen             # force-regenerate the golden files
#   scripts/test.sh --dir build-sync --type Debug --sync-check
#
# WHY THIS EXISTS RATHER THAN A BARE `ctest`
#
# Because `ctest` on a fresh clone passes, and means nothing.
#
# The golden reference files are gitignored generated data, so a fresh clone has none.
# The cpu_ref and kernels suites correctly SKIP when they are absent -- skipping is the
# honest outcome for a missing prerequisite (tests/test_framework.hpp explains why a
# skip beats both a silent pass and a hard failure). But it also means the headline
# result reads `passed 20 skipped 16` and looks fine at a glance, while the tests that
# actually compare numbers against float64 truth never ran.
#
# So this script generates the data if it is missing, and regenerates it if
# tools/gen_reference.py has been modified since the files were written -- a stale
# golden file is the one failure mode that produces a confident wrong answer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="build"
DO_BUILD=1
REGEN=0
BUILD_ARGS=()
CTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir|-B)      BUILD_DIR="$2"; BUILD_ARGS+=(--dir "$2"); shift 2 ;;
    --type|-t)     BUILD_ARGS+=(--type "$2");                shift 2 ;;
    --sync-check)  BUILD_ARGS+=(--sync-check);               shift ;;
    --werror)      BUILD_ARGS+=(--werror);                   shift ;;
    --clean)       BUILD_ARGS+=(--clean);                    shift ;;
    --no-build)    DO_BUILD=0;                               shift ;;
    --regen)       REGEN=1;                                  shift ;;
    -R|--filter)   CTEST_ARGS+=(-R "$2");                    shift 2 ;;
    -h|--help)     sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
    *)             CTEST_ARGS+=("$1");                       shift ;;
  esac
done

if [[ $DO_BUILD -eq 1 ]]; then
  "$REPO_ROOT/scripts/build.sh" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
fi

if [[ ! -e "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo "test.sh: $BUILD_DIR is not configured. Run without --no-build." >&2
  exit 2
fi

#-----------------------------------------------------------------------------------
# Reference data
#-----------------------------------------------------------------------------------
GOLDEN_DIR="tests/golden"
GENERATOR="tools/gen_reference.py"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=""
fi

need_reference=0
if [[ $REGEN -eq 1 ]]; then
  need_reference=1
  echo "==> --regen: regenerating reference data"
elif ! compgen -G "$GOLDEN_DIR/*.bin" >/dev/null; then
  need_reference=1
  echo "==> no reference data in $GOLDEN_DIR/"
elif [[ -z "$(find "$GOLDEN_DIR" -name '*.bin' -newer "$GENERATOR" -print -quit)" ]]; then
  # Every golden file predates the generator, so the generator has changed since they
  # were written and they may no longer be what it would produce.
  need_reference=1
  echo "==> $GENERATOR is newer than every golden file -- regenerating"
fi

if [[ $need_reference -eq 1 ]]; then
  if [[ -z "$PYTHON" ]]; then
    echo "test.sh: no python3 on PATH, cannot generate reference data." >&2
    echo "         The cpu_ref and kernels suites will skip. Install Python 3 and" >&2
    echo "         either numpy or torch, then re-run." >&2
  else
    "$PYTHON" "$GENERATOR"
  fi
fi

#-----------------------------------------------------------------------------------
# Test
#-----------------------------------------------------------------------------------
echo "==> ctest"
ctest --test-dir "$BUILD_DIR" --output-on-failure \
      ${CTEST_ARGS[@]+"${CTEST_ARGS[@]}"}

# A reminder that a green run on a CPU-only machine says nothing about the kernels.
# The kernels suite is not failing there -- it is absent, which is a different thing
# and much easier to overlook. docs/01-dev-environment.md makes the same point.
#
# The generated config header is the authority on which configuration this is.
# CMakeCache.txt is not: check_language(CUDA) writes CMAKE_CUDA_COMPILER into the
# cache either way, as a path or as NOTFOUND, so its mere presence proves nothing.
CONFIG_HPP="$BUILD_DIR/generated/engine/config.hpp"
if [[ -f "$CONFIG_HPP" ]] && ! grep -q '^#define ENGINE_CUDA_ENABLED 1' "$CONFIG_HPP"; then
  echo
  echo "Note: this is a CPU-only build. The kernels suite was not built, not skipped --"
  echo "      it does not exist here. Kernel correctness is established on the GPU box:"
  echo "      commit, push, and run scripts/gpu-run.sh there."
fi
