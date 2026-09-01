// CPU reference: sum reduction.
//
// The double accumulator is a deliberate choice. A float accumulator over a large
// array loses low-order bits progressively (each add rounds to 24 bits of
// mantissa), so summing 1e6 values of similar magnitude in float can drift by a
// relative error of ~1e-3. A GPU tree reduction is actually MORE accurate than a
// sequential float sum, because the tree keeps partial sums of similar magnitude.
//
// If the oracle used float, a correct GPU kernel would appear to "fail" against it.
// Accumulating in double sidesteps the argument entirely.

#include <engine/cpu_ref.hpp>

namespace engine::cpu {

double reduce_sum(const float* x, std::int64_t n) {
  double acc = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    acc += static_cast<double>(x[i]);
  }
  return acc;
}

}  // namespace engine::cpu
