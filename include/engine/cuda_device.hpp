#pragma once
//===----------------------------------------------------------------------===//
// engine/cuda_device.hpp -- device discovery and reporting.
//
// Small file, but it earns its place: it prints the numbers you need in order to
// reason about occupancy and shared-memory budgets, and it is what tells you
// empirically which architecture you are actually running on.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#include <string>

namespace engine {

/// Human-readable one-line summary of the current device, e.g.
///   "NVIDIA GeForce RTX 4090 | sm_89 | 23.6 GiB | 128 SMs | 99 KiB shared/block | 1008 GB/s"
/// Returns "no CUDA device" on CPU-only builds. Safe to call unconditionally --
/// benchmark output should always record the hardware it ran on.
std::string cuda_device_summary();

/// Prints cuda_device_summary() plus the fuller property dump (warp size, max
/// threads/block, registers/SM, L2 size, clock rates).
void print_cuda_device_info();

/// Number of visible CUDA devices; 0 on CPU-only builds.
int cuda_device_count();

/// Theoretical peak global-memory bandwidth in GB/s, computed from the clock and
/// bus width. Compare a memory-bound kernel's achieved bandwidth against this to
/// know whether it is worth optimising further -- on the RTX 4090 the ceiling is
/// about 1008 GB/s, so a kernel sustaining ~900 GB/s is essentially done and
/// further effort belongs elsewhere.
double cuda_peak_bandwidth_gbs();

/// Driver version string (e.g. "12.6"). Returns "none" on CPU-only builds.
std::string cuda_driver_version();

/// Device clock rate in kHz. Returns 0 on CPU-only builds.
int cuda_clock_rate_khz();

}  // namespace engine
