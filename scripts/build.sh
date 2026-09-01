#!/usr/bin/env bash
#
# Configure and build the engine.
#
#   scripts/build.sh                        # RelWithDebInfo into build/
#   scripts/build.sh --type Debug           # a debug build, same directory
#   scripts/build.sh --dir build-sync --sync-check --type Debug
#   scripts/build.sh --clean                # delete the build directory first
#   scripts/build.sh -- -DENGINE_BUILD_BENCH=OFF     # extra cmake args
#
# Works on Machine A (no CUDA) and Machine B (CUDA). It does not know or care which:
# the build system detects CUDA itself and reports what it found. See
# docs/adr/0001-cuda-optional-build.md.
#
# WHY A SCRIPT AND NOT JUST THE TWO CMAKE COMMANDS
#
# Three things here are easy to get wrong by hand, and two of them fail silently:
#
#   1. `cmake --build build -j` with no number after -j. Harmless with Ninja, which
#      defaults to a sensible job count. With Make it means UNLIMITED parallelism --
#      one compile process per translation unit, all at once. nvcc peaks around 1-2 GB
#      of RAM per file, so this is how you get an out-of-memory kill or a machine that
#      stops responding. This script always passes an explicit count.
#
#   2. Switching generators on an existing build directory is a hard CMake error with
#      a confusing message. So -G is passed only when creating a directory, never when
#      reusing one.
#
#   3. Forgetting that a build directory remembers its configure flags. A `build/`
#      configured once with -DCMAKE_BUILD_TYPE=Debug stays Debug forever unless you
#      say otherwise, and a debug build silently invalidates every benchmark number.
#      This script re-runs configure every time -- it takes about a second, and it
#      means the flags you passed are the flags in effect.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_TYPE="RelWithDebInfo"
BUILD_DIR="build"
SYNC_CHECK="OFF"
WERROR="OFF"
CLEAN=0
CMAKE_EXTRA=()

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type|-t)     BUILD_TYPE="$2"; shift 2 ;;
    --dir|-B)      BUILD_DIR="$2";  shift 2 ;;
    --sync-check)  SYNC_CHECK="ON"; shift ;;
    --werror)      WERROR="ON";     shift ;;
    --clean)       CLEAN=1;         shift ;;
    -h|--help)     usage 0 ;;
    --)            shift; CMAKE_EXTRA=("$@"); break ;;
    *) echo "build.sh: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done

case "$BUILD_TYPE" in
  Debug|Release|RelWithDebInfo|MinSizeRel) ;;
  *) echo "build.sh: unknown build type '$BUILD_TYPE'" >&2; exit 2 ;;
esac

# Benchmarking a Debug build measures the debug build. Say so now rather than letting
# a table of meaningless numbers reach the lab notebook; bench_kernels also prints the
# build type in its footer as a second line of defence.
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  echo "build.sh: NOTE -- Debug build. Correct for debugging, useless for timing." >&2
fi
if [[ "$SYNC_CHECK" == "ON" ]]; then
  echo "build.sh: NOTE -- sync-check ON. Every kernel launch synchronises, so any" >&2
  echo "          benchmark from this directory is a measurement of cudaDeviceSynchronize." >&2
fi

if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
  echo "==> removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# Job count. `nproc` on Linux, `sysctl` on macOS, and a conservative default if
# neither exists -- never an unbounded -j.
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS=4
fi

# Generator: Ninja when creating a fresh directory, and whatever the directory
# already uses when reusing one. Ninja is worth preferring here because it tracks
# header dependencies correctly for .cu files, which the Makefile generator has
# historically been shakier about.
GENERATOR_ARGS=()
if [[ ! -e "$BUILD_DIR/CMakeCache.txt" ]] && command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

echo "==> configuring ($BUILD_TYPE) in $BUILD_DIR"
cmake -B "$BUILD_DIR" -S . \
  "${GENERATOR_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DENGINE_SYNC_CHECK_KERNELS="$SYNC_CHECK" \
  -DENGINE_WERROR="$WERROR" \
  ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}

echo "==> building with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "==> built into $BUILD_DIR/bin"
