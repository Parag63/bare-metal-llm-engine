//===----------------------------------------------------------------------===//
// src/cpu_ref/attention_cpu.cpp -- CPU reference oracles for FP16 MHA/GQA.
//
// Implements two independent FP64 host oracles per ADR-0012:
//   1. attention():        Clear two-pass softmax oracle.
//   2. attention_online(): Scalar tiled online-softmax recurrence oracle
//                          (mirroring FlashAttention-2's rescaling invariant).
//===----------------------------------------------------------------------===//

#include <engine/check.hpp>
#include <engine/cpu_ref.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace engine::cpu {

namespace {

inline std::size_t to_usize(std::int64_t n) { return static_cast<std::size_t>(n); }

/// Validates attention contracts (ADR-0012) and handles zero-length early exits.
/// Returns true if kernel loop should run, false after handling s_q==0 or s_kv==0.
bool validate_and_handle_empty(const half* q, const half* k, const half* v, half* out,
                               std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                               std::int64_t n_kv_heads, std::int64_t head_dim,
                               std::int64_t kv_head_stride, float scale,
                               std::int64_t q_pos_offset, float* out_lse,
                               const char* name) {
  ENGINE_CHECK(q != nullptr, std::string(name) + ": q is null");
  ENGINE_CHECK(k != nullptr, std::string(name) + ": k is null");
  ENGINE_CHECK(v != nullptr, std::string(name) + ": v is null");
  ENGINE_CHECK(out != nullptr, std::string(name) + ": out is null");
  ENGINE_CHECK(s_q >= 0, std::string(name) + ": s_q must be non-negative");
  ENGINE_CHECK(s_kv >= 0, std::string(name) + ": s_kv must be non-negative");
  ENGINE_CHECK(q_pos_offset >= 0,
               std::string(name) + ": q_pos_offset must be non-negative");
  ENGINE_CHECK(n_heads > 0, std::string(name) + ": n_heads must be positive");
  ENGINE_CHECK(n_kv_heads > 0, std::string(name) + ": n_kv_heads must be positive");
  ENGINE_CHECK(n_heads % n_kv_heads == 0,
               std::string(name) + ": n_heads must be divisible by n_kv_heads");
  ENGINE_CHECK(head_dim == 64 || head_dim == 128,
               std::string(name) + ": head_dim must be 64 or 128");
  ENGINE_CHECK(std::isfinite(scale), std::string(name) + ": scale must be finite");
  const std::int64_t min_kv_stride = (s_kv > 0) ? (s_kv * head_dim) : 0;
  ENGINE_CHECK(kv_head_stride >= min_kv_stride,
               std::string(name) + ": kv_head_stride < s_kv * head_dim");

  if (s_q == 0) return false;

  if (s_kv == 0) {
    const std::size_t total_out = to_usize(s_q) * to_usize(n_heads) * to_usize(head_dim);
    std::fill_n(out, total_out, float_to_half(0.0f));
    if (out_lse != nullptr) {
      std::fill_n(out_lse, to_usize(s_q) * to_usize(n_heads),
                  -std::numeric_limits<float>::infinity());
    }
    return false;
  }
  return true;
}

inline void write_zero_row(half* out_row, std::int64_t head_dim, float* out_lse,
                           std::int64_t row_idx) {
  std::fill_n(out_row, to_usize(head_dim), float_to_half(0.0f));
  if (out_lse != nullptr) {
    out_lse[to_usize(row_idx)] = -std::numeric_limits<float>::infinity();
  }
}

inline double qk_dot_f64(const half* q_row, const half* k_row, std::int64_t head_dim) {
  double dot = 0.0;
  for (std::int64_t d = 0; d < head_dim; ++d) {
    dot += static_cast<double>(half_to_float(q_row[d])) *
           static_cast<double>(half_to_float(k_row[d]));
  }
  return dot;
}

}  // namespace

void attention(const half* q, const half* k, const half* v, half* out, std::int64_t s_q,
               std::int64_t s_kv, std::int64_t n_heads, std::int64_t n_kv_heads,
               std::int64_t head_dim, std::int64_t kv_head_stride, float scale,
               std::int64_t q_pos_offset, bool is_causal, float* out_lse) {
  if (!validate_and_handle_empty(q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim,
                                 kv_head_stride, scale, q_pos_offset, out_lse,
                                 "cpu::attention")) {
    return;
  }

  const std::int64_t group_size = n_heads / n_kv_heads;
  const double scale_d = static_cast<double>(scale);
  std::vector<double> scores(to_usize(s_kv));
  std::vector<double> acc(to_usize(head_dim));

  for (std::int64_t i = 0; i < s_q; ++i) {
    const std::int64_t max_j =
        is_causal ? std::min<std::int64_t>(s_kv - 1, q_pos_offset + i) : (s_kv - 1);

    for (std::int64_t h = 0; h < n_heads; ++h) {
      const std::int64_t row_idx = i * n_heads + h;
      const half* q_row = q + to_usize(row_idx) * to_usize(head_dim);
      half* out_row = out + to_usize(row_idx) * to_usize(head_dim);

      if (max_j < 0) {
        write_zero_row(out_row, head_dim, out_lse, row_idx);
        continue;
      }

      const std::int64_t h_kv = h / group_size;
      const half* k_head = k + to_usize(h_kv) * to_usize(kv_head_stride);
      const half* v_head = v + to_usize(h_kv) * to_usize(kv_head_stride);

      // Pass 1: dot products and row max over valid keys [0, max_j].
      double m = -std::numeric_limits<double>::infinity();
      for (std::int64_t j = 0; j <= max_j; ++j) {
        const double s = qk_dot_f64(q_row, k_head + j * head_dim, head_dim) * scale_d;
        scores[to_usize(j)] = s;
        if (s > m) m = s;
      }

      if (!std::isfinite(m) && m < 0.0) {
        write_zero_row(out_row, head_dim, out_lse, row_idx);
        continue;
      }

      // Pass 2: shifted exponentials, sum, and PV accumulation in FP64.
      double sum_exp = 0.0;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (std::int64_t j = 0; j <= max_j; ++j) {
        const double w = std::exp(scores[to_usize(j)] - m);
        sum_exp += w;
        const half* v_row = v_head + j * head_dim;
        for (std::int64_t d = 0; d < head_dim; ++d) {
          acc[to_usize(d)] += w * static_cast<double>(half_to_float(v_row[d]));
        }
      }

      if (sum_exp == 0.0) {
        write_zero_row(out_row, head_dim, out_lse, row_idx);
        continue;
      }

      const double inv_sum = 1.0 / sum_exp;
      for (std::int64_t d = 0; d < head_dim; ++d) {
        out_row[d] = float_to_half(static_cast<float>(acc[to_usize(d)] * inv_sum));
      }
      if (out_lse != nullptr) {
        out_lse[to_usize(row_idx)] = static_cast<float>(m + std::log(sum_exp));
      }
    }
  }
}

void attention_online(const half* q, const half* k, const half* v, half* out,
                      std::int64_t s_q, std::int64_t s_kv, std::int64_t n_heads,
                      std::int64_t n_kv_heads, std::int64_t head_dim,
                      std::int64_t kv_head_stride, float scale, std::int64_t q_pos_offset,
                      bool is_causal, float* out_lse) {
  if (!validate_and_handle_empty(q, k, v, out, s_q, s_kv, n_heads, n_kv_heads, head_dim,
                                 kv_head_stride, scale, q_pos_offset, out_lse,
                                 "cpu::attention_online")) {
    return;
  }

  constexpr std::int64_t kTileKv = 16;
  const std::int64_t group_size = n_heads / n_kv_heads;
  const double scale_d = static_cast<double>(scale);
  double tile_scores[kTileKv];
  std::vector<double> acc(to_usize(head_dim));

  for (std::int64_t i = 0; i < s_q; ++i) {
    const std::int64_t max_j =
        is_causal ? std::min<std::int64_t>(s_kv - 1, q_pos_offset + i) : (s_kv - 1);

    for (std::int64_t h = 0; h < n_heads; ++h) {
      const std::int64_t row_idx = i * n_heads + h;
      const half* q_row = q + to_usize(row_idx) * to_usize(head_dim);
      half* out_row = out + to_usize(row_idx) * to_usize(head_dim);

      if (max_j < 0) {
        write_zero_row(out_row, head_dim, out_lse, row_idx);
        continue;
      }

      const std::int64_t h_kv = h / group_size;
      const half* k_head = k + to_usize(h_kv) * to_usize(kv_head_stride);
      const half* v_head = v + to_usize(h_kv) * to_usize(kv_head_stride);

      double m = -std::numeric_limits<double>::infinity();
      double l = 0.0;
      std::fill(acc.begin(), acc.end(), 0.0);

      for (std::int64_t j0 = 0; j0 <= max_j; j0 += kTileKv) {
        const std::int64_t j_end = std::min<std::int64_t>(max_j + 1, j0 + kTileKv);
        double m_block = -std::numeric_limits<double>::infinity();
        for (std::int64_t j = j0; j < j_end; ++j) {
          const double s = qk_dot_f64(q_row, k_head + j * head_dim, head_dim) * scale_d;
          tile_scores[j - j0] = s;
          if (s > m_block) m_block = s;
        }

        if (!std::isfinite(m_block) && m_block < 0.0) {
          continue;
        }

        const double m_new = std::max(m, m_block);
        const double alpha =
            (m == -std::numeric_limits<double>::infinity()) ? 0.0 : std::exp(m - m_new);
        l *= alpha;
        for (std::int64_t d = 0; d < head_dim; ++d) {
          acc[to_usize(d)] *= alpha;
        }

        for (std::int64_t j = j0; j < j_end; ++j) {
          const double p = std::exp(tile_scores[j - j0] - m_new);
          l += p;
          const half* v_row = v_head + j * head_dim;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            acc[to_usize(d)] += p * static_cast<double>(half_to_float(v_row[d]));
          }
        }
        m = m_new;
      }

      if (l == 0.0) {
        write_zero_row(out_row, head_dim, out_lse, row_idx);
        continue;
      }

      const double inv_l = 1.0 / l;
      for (std::int64_t d = 0; d < head_dim; ++d) {
        out_row[d] = float_to_half(static_cast<float>(acc[to_usize(d)] * inv_l));
      }
      if (out_lse != nullptr) {
        out_lse[to_usize(row_idx)] = static_cast<float>(m + std::log(l));
      }
    }
  }
}

}  // namespace engine::cpu
