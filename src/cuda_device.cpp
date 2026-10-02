//===----------------------------------------------------------------------===//
// src/cuda_device.cpp -- device discovery.
//
// Compiled in BOTH configurations. On a CPU-only build every function returns a
// harmless empty answer, so callers (benchmarks especially) never need their own
// #if guards.
//
// This file is also how you empirically confirm what hardware you are on. Do not
// trust a spec sheet; print cudaDeviceProp.
//===----------------------------------------------------------------------===//

#include <engine/cuda_device.hpp>

#include <engine/check.hpp>
#include <engine/config.hpp>

#include <cstdio>
#include <sstream>

#if ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace engine {

#if !ENGINE_HAS_CUDA

std::string cuda_device_summary() { return "no CUDA device (CPU-only build)"; }
int cuda_device_count() { return 0; }
double cuda_peak_bandwidth_gbs() { return 0.0; }
std::string cuda_driver_version() { return "none"; }
int cuda_clock_rate_khz() { return 0; }
int cuda_live_sm_clock_mhz() { return 0; }
int cuda_live_mem_clock_mhz() { return 0; }
void print_cuda_device_info() {
  std::printf("CUDA: not available -- this is a CPU-only build.\n");
}

#else

int cuda_device_count() {
  int n = 0;
  // Deliberately NOT CUDA_CHECK: on a machine with no driver this returns an error
  // rather than 0, and "how many GPUs are there" should answer zero, not throw.
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

namespace {
bool get_props(cudaDeviceProp& prop) {
  if (cuda_device_count() == 0) return false;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  return cudaGetDeviceProperties(&prop, dev) == cudaSuccess;
}
}  // namespace

double cuda_peak_bandwidth_gbs() {
  cudaDeviceProp prop{};
  if (!get_props(prop)) return 0.0;

  // memoryClockRate is in kHz, memoryBusWidth in bits.
  //   bytes/s = clock_Hz * (bus_bits / 8) * 2      <- 2 for double data rate
  // Dividing by 1e9 gives GB/s in the decimal sense, which is how NVIDIA quotes it.
  const double clock_hz = static_cast<double>(prop.memoryClockRate) * 1.0e3;
  const double bus_bytes = static_cast<double>(prop.memoryBusWidth) / 8.0;
  return clock_hz * bus_bytes * 2.0 / 1.0e9;
}

std::string cuda_driver_version() {
  int drv = 0;
  if (cudaDriverGetVersion(&drv) != cudaSuccess) return "unknown";
  return std::to_string(drv / 1000) + "." + std::to_string((drv % 100) / 10);
}

int cuda_clock_rate_khz() {
  cudaDeviceProp prop{};
  if (!get_props(prop)) return 0;
  return prop.clockRate;
}

static int cuda_live_clock_mhz(int clock_type) {
  if (cuda_device_count() == 0) return 0;

  using nvmlReturn_t = int;
  using nvmlDevice_t = void*;

  using pfn_nvmlInit_v2 = nvmlReturn_t (*)();
  using pfn_nvmlDeviceGetHandleByIndex_v2 = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
  using pfn_nvmlDeviceGetClockInfo = nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int*);
  using pfn_nvmlShutdown = nvmlReturn_t (*)();

#if defined(_WIN32)
  HMODULE lib = LoadLibraryA("nvml.dll");
  if (!lib) return (clock_type == 1) ? (cuda_clock_rate_khz() / 1000) : 0;
  auto fn_init = reinterpret_cast<pfn_nvmlInit_v2>(GetProcAddress(lib, "nvmlInit_v2"));
  auto fn_get_handle = reinterpret_cast<pfn_nvmlDeviceGetHandleByIndex_v2>(
      GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
  auto fn_get_clock = reinterpret_cast<pfn_nvmlDeviceGetClockInfo>(
      GetProcAddress(lib, "nvmlDeviceGetClockInfo"));
  auto fn_shutdown =
      reinterpret_cast<pfn_nvmlShutdown>(GetProcAddress(lib, "nvmlShutdown"));
#else
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
  if (!lib) return (clock_type == 1) ? (cuda_clock_rate_khz() / 1000) : 0;
  auto fn_init = reinterpret_cast<pfn_nvmlInit_v2>(dlsym(lib, "nvmlInit_v2"));
  auto fn_get_handle = reinterpret_cast<pfn_nvmlDeviceGetHandleByIndex_v2>(
      dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
  auto fn_get_clock =
      reinterpret_cast<pfn_nvmlDeviceGetClockInfo>(dlsym(lib, "nvmlDeviceGetClockInfo"));
  auto fn_shutdown = reinterpret_cast<pfn_nvmlShutdown>(dlsym(lib, "nvmlShutdown"));
#endif

  int result_clock = 0;
  if (fn_init && fn_get_handle && fn_get_clock) {
    if (fn_init() == 0) {
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        nvmlDevice_t handle = nullptr;
        if (fn_get_handle(static_cast<unsigned int>(dev), &handle) == 0) {
          unsigned int clock_mhz = 0;
          if (fn_get_clock(handle, clock_type, &clock_mhz) == 0) {
            result_clock = static_cast<int>(clock_mhz);
          }
        }
      }
      if (fn_shutdown) fn_shutdown();
    }
  }

#if defined(_WIN32)
  FreeLibrary(lib);
#else
  dlclose(lib);
#endif

  return result_clock;
}

int cuda_live_sm_clock_mhz() {
  constexpr int kNvmlClockSm = 1;
  int clk = cuda_live_clock_mhz(kNvmlClockSm);
  if (clk > 0) return clk;
  return cuda_clock_rate_khz() / 1000;
}

int cuda_live_mem_clock_mhz() {
  constexpr int kNvmlClockMem = 2;
  return cuda_live_clock_mhz(kNvmlClockMem);
}

std::string cuda_device_summary() {
  cudaDeviceProp prop{};
  if (!get_props(prop)) return "no CUDA device";

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(1);
  oss << prop.name << " | sm_" << prop.major << prop.minor << " | "
      << (static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)) << " GiB"
      << " | " << prop.multiProcessorCount << " SMs" << " | "
      << (prop.sharedMemPerBlockOptin / 1024) << " KiB shared/block" << " | "
      << cuda_peak_bandwidth_gbs() << " GB/s";
  return oss.str();
}

void print_cuda_device_info() {
  if (cuda_device_count() == 0) {
    std::printf("CUDA: built with support, but no device is visible.\n");
    return;
  }

  cudaDeviceProp prop{};
  if (!get_props(prop)) {
    std::printf("CUDA: failed to query device properties.\n");
    return;
  }

  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);

  std::printf("=========================== CUDA device ===========================\n");
  std::printf("  Name                      : %s\n", prop.name);
  std::printf("  Compute capability        : sm_%d%d\n", prop.major, prop.minor);
  std::printf("  Compiled for              : sm_%s\n", ENGINE_CUDA_ARCH_STRING);
  std::printf("  Runtime / driver version  : %d / %d\n", rt, drv);
  std::printf("  Global memory             : %.2f GiB\n",
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Peak bandwidth (theor.)   : %.0f GB/s\n", cuda_peak_bandwidth_gbs());
  std::printf("  Multiprocessors (SMs)     : %d\n", prop.multiProcessorCount);
  std::printf("  Warp size                 : %d\n", prop.warpSize);
  std::printf("  Max threads / block       : %d\n", prop.maxThreadsPerBlock);
  std::printf("  Max threads / SM          : %d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("  Shared mem / block        : %zu B\n", prop.sharedMemPerBlock);
  std::printf("  Shared mem / block optin  : %zu B\n",
              static_cast<size_t>(prop.sharedMemPerBlockOptin));
  std::printf("  Shared mem / SM           : %zu B\n", prop.sharedMemPerMultiprocessor);
  std::printf("  Registers / block         : %d\n", prop.regsPerBlock);
  std::printf("  L2 cache                  : %d KiB\n", prop.l2CacheSize / 1024);
  const int live_clk = cuda_live_sm_clock_mhz();
  if (live_clk > 0) {
    std::printf(
        "  Clock rate                : %d MHz (live SM via NVML; nominal %.0f MHz)\n",
        live_clk, prop.clockRate / 1000.0);
  } else {
    std::printf("  Clock rate                : %.0f MHz\n", prop.clockRate / 1000.0);
  }
  const int live_mem_clk = cuda_live_mem_clock_mhz();
  if (live_mem_clk > 0) {
    std::printf("  Memory clock rate         : %d MHz (live via NVML)\n", live_mem_clk);
  }
  std::printf("  Async engines             : %d\n", prop.asyncEngineCount);
  std::printf("  Concurrent kernels        : %s\n",
              prop.concurrentKernels ? "yes" : "no");
  std::printf("===================================================================\n");

  // Loud, actionable mismatch warning. Building for the wrong architecture either
  // fails to launch or silently falls back to JIT-compiled PTX, which is slower
  // and would quietly corrupt your benchmark numbers.
  std::ostringstream actual;
  actual << prop.major << prop.minor;
  if (actual.str() != std::string(ENGINE_CUDA_ARCH_STRING)) {
    std::printf(
        "\n  WARNING: this binary was compiled for sm_%s but the device is sm_%s.\n"
        "           Reconfigure with -DENGINE_CUDA_ARCH=%s for correct, fast code.\n\n",
        ENGINE_CUDA_ARCH_STRING, actual.str().c_str(), actual.str().c_str());
  }
}

#endif  // ENGINE_HAS_CUDA

}  // namespace engine
