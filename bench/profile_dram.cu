//===----------------------------------------------------------------------===//
// bench/profile_dram.cu -- Profile DRAM traffic of individual fused/unfused kernels.
//
// Used with NVIDIA Nsight Compute (ncu) to measure exact dram__bytes.sum
// against theoretical 8 MiB and 16 MiB DRAM reduction predictions.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>

#if !ENGINE_HAS_CUDA
#include <cstdio>
int main() {
  std::printf("profile_dram requires CUDA.\n");
  return 0;
}
#else

#include <engine/check.hpp>
#include <engine/device_buffer.hpp>
#include <engine/kernels.hpp>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using engine::DeviceBuffer;

std::vector<float> random_host(std::size_t n, unsigned seed = 20261001u) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n);
  for (float& x : v) x = dist(gen);
  return v;
}

// Flush L2 cache by clearing a 128 MiB buffer
void flush_l2(DeviceBuffer<float>& l2_buf) {
  CUDA_CHECK(cudaMemset(l2_buf.get(), 0, l2_buf.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("Usage: %s <case_name>\n", argv[0]);
    std::printf("Available cases:\n");
    std::printf("  residual_separate_512\n");
    std::printf("  residual_fused_512\n");
    std::printf("  rmsnorm_linear_separate_512\n");
    std::printf("  rmsnorm_linear_fused_512\n");
    std::printf("  rmsnorm_linear_separate_1\n");
    std::printf("  rmsnorm_linear_fused_1\n");
    return 1;
  }

  const std::string name = argv[1];

  // 128 MiB L2 eviction buffer (exceeds 48 MiB L2 cache on RTX 4070 SUPER)
  DeviceBuffer<float> l2_evict(32 * 1024 * 1024);

  if (name == "residual_separate_512") {
    const std::int64_t rows = 512, cols = 4096;
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    DeviceBuffer<float> x(random_host(n, 1u));
    DeviceBuffer<float> res(random_host(n, 2u));
    DeviceBuffer<float> weight(random_host(cols, 3u));
    DeviceBuffer<float> temp_sum(n);
    DeviceBuffer<float> norm_out(n);

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Single timed pass
    engine::cuda::vector_add(x.get(), res.get(), temp_sum.get(), rows * cols);
    engine::cuda::rmsnorm(temp_sum.get(), weight.get(), norm_out.get(), rows, cols,
                          1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed residual_separate_512\n");

  } else if (name == "residual_fused_512") {
    const std::int64_t rows = 512, cols = 4096;
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    DeviceBuffer<float> x(random_host(n, 1u));
    DeviceBuffer<float> res(random_host(n, 2u));
    DeviceBuffer<float> weight(random_host(cols, 3u));
    DeviceBuffer<float> norm_out(n);
    DeviceBuffer<float> sum_out(n);

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::residual_rmsnorm(x.get(), res.get(), weight.get(), norm_out.get(),
                                   sum_out.get(), rows, cols, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed residual_fused_512\n");

  } else if (name == "rmsnorm_linear_separate_512") {
    const std::int64_t M = 512, N = 4096, K = 4096;
    DeviceBuffer<float> in(random_host(static_cast<std::size_t>(M * K), 1u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(K), 2u));
    DeviceBuffer<float> W(random_host(static_cast<std::size_t>(K * N), 3u));
    DeviceBuffer<float> temp(static_cast<std::size_t>(M * K));
    DeviceBuffer<float> out(static_cast<std::size_t>(M * N));

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::rmsnorm(in.get(), weight.get(), temp.get(), M, K, 1e-5f);
    engine::cuda::matmul_tiled(temp.get(), W.get(), out.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed rmsnorm_linear_separate_512\n");

  } else if (name == "rmsnorm_linear_fused_512") {
    const std::int64_t M = 512, N = 4096, K = 4096;
    DeviceBuffer<float> in(random_host(static_cast<std::size_t>(M * K), 1u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(K), 2u));
    DeviceBuffer<float> W(random_host(static_cast<std::size_t>(K * N), 3u));
    DeviceBuffer<float> out(static_cast<std::size_t>(M * N));

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::rmsnorm_linear_fused_direct(in.get(), weight.get(), W.get(), out.get(),
                                              M, N, K, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed rmsnorm_linear_fused_512\n");

  } else if (name == "rmsnorm_linear_separate_1") {
    const std::int64_t M = 1, N = 4096, K = 4096;
    DeviceBuffer<float> in(random_host(static_cast<std::size_t>(M * K), 1u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(K), 2u));
    DeviceBuffer<float> W(random_host(static_cast<std::size_t>(K * N), 3u));
    DeviceBuffer<float> temp(static_cast<std::size_t>(M * K));
    DeviceBuffer<float> out(static_cast<std::size_t>(M * N));

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::rmsnorm(in.get(), weight.get(), temp.get(), M, K, 1e-5f);
    engine::cuda::matmul_tiled(temp.get(), W.get(), out.get(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed rmsnorm_linear_separate_1\n");

  } else if (name == "rmsnorm_linear_fused_1") {
    const std::int64_t M = 1, N = 4096, K = 4096;
    DeviceBuffer<float> in(random_host(static_cast<std::size_t>(M * K), 1u));
    DeviceBuffer<float> weight(random_host(static_cast<std::size_t>(K), 2u));
    DeviceBuffer<float> W(random_host(static_cast<std::size_t>(K * N), 3u));
    DeviceBuffer<float> out(static_cast<std::size_t>(M * N));

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::rmsnorm_linear_fused_direct(in.get(), weight.get(), W.get(), out.get(),
                                              M, N, K, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed rmsnorm_linear_fused_1\n");

  } else if (name == "swiglu_512") {
    const std::int64_t n = 512 * 11008;
    DeviceBuffer<float> gate(random_host(static_cast<std::size_t>(n), 1u));
    DeviceBuffer<float> up(random_host(static_cast<std::size_t>(n), 2u));
    DeviceBuffer<float> out(static_cast<std::size_t>(n));

    flush_l2(l2_evict);
    CUDA_CHECK(cudaDeviceSynchronize());

    engine::cuda::swiglu(gate.get(), up.get(), out.get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Executed swiglu_512\n");

  } else {
    std::fprintf(stderr, "Unknown case: %s\n", name.c_str());
    return 2;
  }

  return 0;
}
#endif
