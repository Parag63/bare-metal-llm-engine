#!/usr/bin/env bash
# scripts/profile_kernels.sh -- Profile key kernels with NVIDIA Nsight Compute.
#
# Generates .ncu-rep reports and extracts key metrics (DRAM bytes, achieved occupancy,
# warp stall reasons) to validate architectural predictions.
#
# Prerequisites:
#   1. Built with -DCMAKE_BUILD_TYPE=RelWithDebInfo (keeps symbols and -lineinfo)
#   2. GPU performance counter permissions enabled in NVIDIA Control Panel:
#      Developer -> Manage GPU Performance Counters -> Allow access to all users
#      (or run as administrator)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILES_DIR="${REPO_ROOT}/docs/profiles"
BENCH_BIN="${REPO_ROOT}/build/bin/bench_kernels"

mkdir -p "${PROFILES_DIR}"

NCU="$(which ncu || echo "/usr/local/cuda/bin/ncu")"
if [ ! -x "${NCU}" ]; then
  echo "Error: ncu (Nsight Compute) not found." >&2
  exit 1
fi

echo "=== NVIDIA Nsight Compute Profiler ==="
echo "Profiler binary : ${NCU}"
echo "Output directory: ${PROFILES_DIR}"
echo ""

# 1. Profile matmul_tiled (4096^3)
echo "[1/4] Profiling matmul_tiled (sm_89 occupancy & DRAM throughput)..."
"${NCU}" --set full \
  --kernel-name "engine::cuda::matmul_tiled" \
  --launch-count 1 \
  --export "${PROFILES_DIR}/matmul_tiled_4096" \
  "${BENCH_BIN}" --quick || true

# 2. Profile GEMV (decode M=1)
echo "[2/4] Profiling gemv (decode M=1, DRAM streaming bandwidth)..."
"${NCU}" --metrics dram__bytes.sum,sm__throughput.avg_pct_of_peak_sustained_elapsed,launch__occupancy \
  --kernel-name "engine::cuda::gemv_fast_k8_n64" \
  --launch-count 1 \
  --export "${PROFILES_DIR}/gemv_decode" \
  "${BENCH_BIN}" --quick || true

# 3. Profile rmsnorm_linear (fused vs separate DRAM traffic validation)
echo "[3/4] Profiling rmsnorm_linear (validating predicted 16 MiB DRAM elimination)..."
"${NCU}" --metrics dram__bytes.sum,gpu__time_duration.sum \
  --kernel-name "engine::cuda::rmsnorm_linear" \
  --launch-count 1 \
  --export "${PROFILES_DIR}/rmsnorm_linear_fused" \
  "${BENCH_BIN}" --quick || true

# 4. Profile residual_rmsnorm (validating predicted 8 MiB DRAM elimination)
echo "[4/4] Profiling residual_rmsnorm..."
"${NCU}" --metrics dram__bytes.sum,gpu__time_duration.sum \
  --kernel-name "engine::cuda::residual_rmsnorm" \
  --launch-count 1 \
  --export "${PROFILES_DIR}/residual_rmsnorm_fused" \
  "${BENCH_BIN}" --quick || true

echo ""
echo "Profiling complete. Profiles saved to ${PROFILES_DIR}/"
