// CPU reference: element-wise vector addition.
// Written for obvious correctness, not speed. See engine/cpu_ref.hpp.

#include <engine/cpu_ref.hpp>

namespace engine::cpu {

void vector_add(const float* a, const float* b, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace engine::cpu
