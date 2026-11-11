//===----------------------------------------------------------------------===//
// src/cpu_ref/rope_cpu.cpp -- CPU reference oracle for Rotary Position Embedding.
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/cpu_ref.hpp>

#include <cmath>
#include <cstddef>

namespace engine::cpu {

namespace {

inline std::size_t to_usize(std::int64_t n) { return static_cast<std::size_t>(n); }

void apply_rope_tensor(half* x, const float* cos_table, const float* sin_table,
                       std::int64_t s, std::int64_t num_heads, std::int64_t head_dim,
                       std::int64_t pos_offset, RopeConvention convention) {
  const std::int64_t half_d = head_dim / 2;
  for (std::int64_t t = 0; t < s; ++t) {
    const std::int64_t pos = pos_offset + t;
    const float* c_row = cos_table + to_usize(pos) * to_usize(half_d);
    const float* s_row = sin_table + to_usize(pos) * to_usize(half_d);

    for (std::int64_t h = 0; h < num_heads; ++h) {
      half* vec =
          x + (to_usize(t) * to_usize(num_heads) + to_usize(h)) * to_usize(head_dim);
      for (std::int64_t i = 0; i < half_d; ++i) {
        const std::size_t idx0 =
            (convention == RopeConvention::RotateHalf) ? to_usize(i) : to_usize(2 * i);
        const std::size_t idx1 = (convention == RopeConvention::RotateHalf)
                                     ? to_usize(i + half_d)
                                     : to_usize(2 * i + 1);

        const double x0 = static_cast<double>(half_to_float(vec[idx0]));
        const double x1 = static_cast<double>(half_to_float(vec[idx1]));
        const double c = static_cast<double>(c_row[to_usize(i)]);
        const double sn = static_cast<double>(s_row[to_usize(i)]);

        vec[idx0] = float_to_half(static_cast<float>(x0 * c - x1 * sn));
        vec[idx1] = float_to_half(static_cast<float>(x0 * sn + x1 * c));
      }
    }
  }
}

}  // namespace

void rope_precompute_tables(float* cos_table, float* sin_table, std::int64_t max_seq_len,
                            std::int64_t head_dim, float theta) {
  ENGINE_CHECK(cos_table != nullptr, "cpu::rope_precompute_tables: cos_table is null");
  ENGINE_CHECK(sin_table != nullptr, "cpu::rope_precompute_tables: sin_table is null");
  ENGINE_CHECK(max_seq_len >= 0,
               "cpu::rope_precompute_tables: max_seq_len must be non-negative");
  ENGINE_CHECK(head_dim > 0 && head_dim % 2 == 0,
               "cpu::rope_precompute_tables: head_dim must be positive and even");
  ENGINE_CHECK(theta > 0.0f && std::isfinite(theta),
               "cpu::rope_precompute_tables: theta must be positive and finite");

  const std::int64_t half_d = head_dim / 2;
  const double theta_d = static_cast<double>(theta);
  const double dim_d = static_cast<double>(head_dim);

  for (std::int64_t pos = 0; pos < max_seq_len; ++pos) {
    const double pos_d = static_cast<double>(pos);
    float* c_row = cos_table + to_usize(pos) * to_usize(half_d);
    float* s_row = sin_table + to_usize(pos) * to_usize(half_d);
    for (std::int64_t i = 0; i < half_d; ++i) {
      const double freq = std::pow(theta_d, -2.0 * static_cast<double>(i) / dim_d);
      const double angle = pos_d * freq;
      c_row[to_usize(i)] = static_cast<float>(std::cos(angle));
      s_row[to_usize(i)] = static_cast<float>(std::sin(angle));
    }
  }
}

void rope(half* q, half* k, const float* cos_table, const float* sin_table,
          std::int64_t s, std::int64_t n_heads, std::int64_t n_kv_heads,
          std::int64_t head_dim, std::int64_t pos_offset, RopeConvention convention) {
  ENGINE_CHECK(q != nullptr, "cpu::rope: q is null");
  ENGINE_CHECK(n_kv_heads >= 0, "cpu::rope: n_kv_heads must be non-negative");
  if (n_kv_heads > 0) {
    ENGINE_CHECK(k != nullptr, "cpu::rope: k is null with n_kv_heads > 0");
  }
  ENGINE_CHECK(cos_table != nullptr, "cpu::rope: cos_table is null");
  ENGINE_CHECK(sin_table != nullptr, "cpu::rope: sin_table is null");
  ENGINE_CHECK(s >= 0, "cpu::rope: s must be non-negative");
  ENGINE_CHECK(n_heads > 0, "cpu::rope: n_heads must be positive");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128,
               "cpu::rope: head_dim must be 64 or 128");
  ENGINE_CHECK(pos_offset >= 0, "cpu::rope: pos_offset must be non-negative");

  if (s == 0) return;

  apply_rope_tensor(q, cos_table, sin_table, s, n_heads, head_dim, pos_offset,
                    convention);
  if (n_kv_heads > 0) {
    apply_rope_tensor(k, cos_table, sin_table, s, n_kv_heads, head_dim, pos_offset,
                      convention);
  }
}

}  // namespace engine::cpu
