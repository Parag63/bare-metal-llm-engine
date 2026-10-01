//===----------------------------------------------------------------------===//
// examples/minimal.cpp -- Minimal downstream consumer application.
//
// Demonstrates consuming bare_metal_llm as an installed CMake package
// without any source tree references or internal relative paths.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>
#include <engine/cpu_ref.hpp>
#include <engine/cuda_device.hpp>
#include <engine/dtype.hpp>
#include <engine/tensor.hpp>

#if ENGINE_HAS_CUDA
#include <engine/kernels.hpp>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <vector>

int main() {
  std::cout << "=====================================================\n";
  std::cout << "  bare_metal_llm Downstream Integration Verification \n";
  std::cout << "=====================================================\n";

  std::cout << "Engine version       : 0.1.0\n";
  std::cout << "CUDA support enabled : " << (ENGINE_HAS_CUDA ? "YES" : "NO (CPU reference path)") << "\n";

  // 1. Verify CPU Tensor creation and DType system
  engine::Tensor x({2, 4}, engine::DType::F32, engine::Device::CPU);
  std::cout << "Created Tensor shape : [" << x.shape()[0] << ", " << x.shape()[1] << "]\n";
  std::cout << "Tensor dtype         : " << engine::dtype_name(x.dtype()) << " ("
            << engine::dtype_storage_bytes(x.dtype(), 1) << " bytes/elem)\n";

  // 2. Verify CPU reference math execution
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> c(4, 0.0f);

  engine::cpu::vector_add(a.data(), b.data(), c.data(), 4);
  std::cout << "CPU vector_add result: [" << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << "]\n";

#if ENGINE_HAS_CUDA
  // 3. Verify CUDA device summary and peak bandwidth query
  if (engine::cuda_device_count() > 0) {
    std::cout << "Active CUDA Device   : " << engine::cuda_device_summary() << "\n";
    std::cout << "Theoretical Peak BW  : " << engine::cuda_peak_bandwidth_gbs() << " GB/s\n";
    std::cout << "Live SM Clock (NVML) : " << engine::cuda_live_sm_clock_mhz() << " MHz\n";
  }
#endif

  std::cout << "\n[PASS] Downstream integration verified successfully!\n";
  return 0;
}
