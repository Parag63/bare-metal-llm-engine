#pragma once
//===----------------------------------------------------------------------===//
// bench/bench_harness.hpp -- timing, statistics, and markdown output.
//
// The point of this file is that a performance number you cannot defend is worse
// than no number at all. Every design choice below exists to make the numbers in
// your report survive the question "how do you know?".
//===----------------------------------------------------------------------===//
// WHY CUDA EVENTS AND NOT std::chrono
//
// A kernel launch is ASYNCHRONOUS. `kernel<<<...>>>()` returns as soon as the work
// is queued -- typically in 3-10 microseconds -- and the kernel may not have started,
// let alone finished. So this:
//
//     auto t0 = std::chrono::steady_clock::now();
//     my_kernel<<<grid, block>>>(...);
//     auto t1 = std::chrono::steady_clock::now();      // WRONG
//
// measures launch overhead. On a small kernel it reports a time that is both wrong
// and impressively fast, which is how bogus speedups get published.
//
// You could fix it with cudaDeviceSynchronize() before t1, and that is roughly
// correct, but it charges the kernel for the synchronisation round-trip (a few
// microseconds of driver work) and for anything else the GPU happened to be doing.
//
// cudaEventRecord() instead inserts a timestamp INTO THE STREAM. The GPU records it
// when it reaches that point in the queue, with roughly half-microsecond resolution,
// and cudaEventElapsedTime() reads the difference. What you get is the GPU-side
// duration of the work between the two events and nothing else. That is the number
// that belongs in the report.
//===----------------------------------------------------------------------===//
// WHY WARMUP, AND WHY IT IS NOT OPTIONAL
//
// The first launch of a kernel is unrepresentative for at least four reasons:
//   * the CUDA context may still be initialising (tens of milliseconds);
//   * the module containing the kernel is loaded lazily on first use;
//   * if the binary has no cubin for this GPU, the driver JIT-compiles the PTX --
//     which can take hundreds of milliseconds and is a one-time cost;
//   * the GPU is at idle clocks and takes a few milliseconds to boost.
//
// Discarding the first few iterations removes all four. Never report a single-shot
// measurement of a GPU kernel.
//===----------------------------------------------------------------------===//
// WHY MEDIAN AND MIN, NOT MEAN
//
// Timing noise on a GPU is almost entirely ADDITIVE and one-sided: the OS scheduler,
// the display driver, another process, a thermal or power cap. Nothing ever makes a
// kernel finish faster than the hardware allows. So the distribution has a hard floor
// and a long right tail, and the mean is dragged around by outliers that say nothing
// about the kernel.
//
//   min     -- the closest available estimate of the kernel's intrinsic cost. Best
//              for comparing two implementations of the same thing.
//   median  -- what you would actually observe in a run. Best for reporting.
//   spread  -- (max - min) / median. A quality-of-measurement figure: under ~2% the
//              measurement is trustworthy; over ~10% something else is using the GPU,
//              or the clocks are moving, and the numbers should not be reported.
//
// Printing the spread is the honest part. It is what lets you say "17.3 ms +/- 0.4%"
// rather than "17.3 ms" and hope nobody asks.
//===----------------------------------------------------------------------===//
// LOCK THE CLOCKS BEFORE BENCHMARKING. THIS IS THE BIGGEST SINGLE SOURCE OF NOISE.
//
// A 4090 boosts and throttles continuously based on temperature and power draw. Back
// to back runs of the same kernel can differ by 10-20% for that reason alone, which
// is larger than most optimisations you will be measuring. Since you have exclusive
// access to the machine, fix the clocks:
//
//     nvidia-smi -q -d SUPPORTED_CLOCKS | head -40      # see what is available
//     sudo nvidia-smi -pm 1                             # persistence mode on
//     sudo nvidia-smi -lgc 2520                          # lock graphics clock (MHz)
//     ... run benchmarks ...
//     sudo nvidia-smi -rgc                               # reset when finished
//
// On Windows run the same commands from an Administrator prompt; nvidia-smi.exe lives
// in C:\Windows\System32 (or C:\Program Files\NVIDIA Corporation\NVSMI on older
// drivers). Lock to a clock the card can hold indefinitely rather than its maximum
// boost -- the point is a number you can reproduce next week, not the biggest number.
//
// Record the locked clock in your lab notebook alongside the results. Two benchmark
// runs at different clocks are not comparable, and six months from now you will not
// remember which was which.
//===----------------------------------------------------------------------===//

#include <engine/config.hpp>
#include <engine/cuda_device.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if ENGINE_HAS_CUDA
#include <engine/check.hpp>
#include <cuda_runtime.h>
#endif

namespace engbench {

inline std::string escape_json(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    if (c == '"') o << "\\\"";
    else if (c == '\\') o << "\\\\";
    else if (c == '\b') o << "\\b";
    else if (c == '\f') o << "\\f";
    else if (c == '\n') o << "\\n";
    else if (c == '\r') o << "\\r";
    else if (c == '\t') o << "\\t";
    else if (static_cast<unsigned char>(c) <= 0x1f) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
      o << buf;
    } else {
      o << c;
    }
  }
  return o.str();
}

//===----------------------------------------------------------------------===//
// Statistics over a set of per-iteration timings.
//===----------------------------------------------------------------------===//
struct Stats {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double max_ms = 0.0;
  int reps = 0;

  /// (max - min) / median, as a percentage. See the header note: this is the
  /// measurement-quality figure, and a benchmark that does not report it is asking
  /// to be believed rather than checked.
  double spread_pct() const {
    if (median_ms <= 0.0) return 0.0;
    return 100.0 * (max_ms - min_ms) / median_ms;
  }
};

inline Stats summarise(std::vector<double> samples) {
  Stats s;
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  s.reps = static_cast<int>(samples.size());
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  const std::size_t mid = samples.size() / 2;
  s.median_ms = (samples.size() % 2 == 0) ? 0.5 * (samples[mid - 1] + samples[mid])
                                          : samples[mid];
  return s;
}

//===----------------------------------------------------------------------===//
// Timing.
//
// Templated on the callable rather than taking std::function so that the timed
// region contains a direct call and nothing else. For a kernel launch the difference
// is irrelevant; for the CPU reference benchmarks, where the whole operation may take
// microseconds, an indirect call through std::function is measurable noise.
//===----------------------------------------------------------------------===//

#if ENGINE_HAS_CUDA
/// Times `fn` (which must enqueue GPU work on the default stream) with CUDA events.
///
/// `fn` is called `warmup + reps` times in total; the first `warmup` results are
/// discarded. Each timed iteration is bracketed by its own event pair, so one
/// unlucky iteration cannot contaminate the others -- which is the whole reason for
/// keeping the individual samples instead of timing the loop and dividing.
template <typename F>
Stats time_gpu(F&& fn, int warmup = 5, int reps = 50) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(reps));

  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    // Waits for `stop` to be reached, not for the whole device to drain. On an
    // otherwise idle GPU the two are the same; the narrower wait is still the
    // correct thing to ask for.
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return summarise(std::move(samples));
}
#endif  // ENGINE_HAS_CUDA

/// Times `fn` on the host. Used for the CPU reference implementations, which are the
/// denominator of every "Nx faster" claim -- so they deserve the same care as the
/// GPU measurements rather than a single stopwatch reading.
///
/// steady_clock, not system_clock: system_clock can be stepped backwards by NTP
/// mid-measurement and produce a negative duration.
template <typename F>
Stats time_cpu(F&& fn, int warmup = 2, int reps = 10) {
  using clock = std::chrono::steady_clock;

  for (int i = 0; i < warmup; ++i) fn();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return summarise(std::move(samples));
}

//===----------------------------------------------------------------------===//
// One benchmarked configuration.
//
// `flops` and `bytes` are per ITERATION and are supplied by the caller, because only
// the caller knows the algorithm. Getting them right is most of the value here:
//
//   flops -- count the arithmetic the ALGORITHM requires, not the instructions the
//            kernel happens to execute. A matmul is 2*M*N*K FLOPs (one multiply and
//            one add per term) whether or not your implementation is clever. Counting
//            actual instructions would make a slow kernel look busy.
//
//   bytes -- count the MINIMUM traffic between the GPU and its DRAM: each input read
//            once, each output written once. A naive matmul re-reads its inputs many
//            times over, but charging it for those re-reads would report a fictitious
//            bandwidth above the hardware peak and hide the very inefficiency you are
//            trying to measure. The ideal-traffic convention makes "achieved GB/s"
//            comparable across implementations, and makes tiling show up as the
//            improvement it is.
//===----------------------------------------------------------------------===//
struct Result {
  std::string name;
  std::string size;
  Stats stats;
  double flops = 0.0;
  double bytes = 0.0;
  bool implemented = true;
  std::string note;

  double gflops() const {
    if (stats.median_ms <= 0.0 || flops <= 0.0) return 0.0;
    return flops / (stats.median_ms * 1.0e6);  // FLOP/ms -> GFLOP/s
  }

  double gbps() const {
    if (stats.median_ms <= 0.0 || bytes <= 0.0) return 0.0;
    return bytes / (stats.median_ms * 1.0e6);  // byte/ms -> GB/s
  }

  /// FLOPs per byte of ideal DRAM traffic. This single number tells you which
  /// resource the kernel is fundamentally limited by, BEFORE you measure anything.
  ///
  /// On an RTX 4090 the balance point is around 82 FLOP/byte (~82.6 TFLOP/s of FP32
  /// against ~1008 GB/s). Below it the kernel is memory-bound and the only thing
  /// that matters is traffic and coalescing; above it the kernel is compute-bound and
  /// arithmetic efficiency starts to matter.
  ///
  ///   vector_add   1 FLOP / 12 bytes  = 0.08   -> hopelessly memory-bound. The
  ///                                              ceiling is peak bandwidth; there
  ///                                              is nothing to optimise beyond
  ///                                              achieving it.
  ///   rmsnorm      ~4 FLOP / 8 bytes  = 0.5    -> memory-bound.
  ///   matmul 1024  2*1024^3 / 12MB    = 170    -> compute-bound. Tiling, register
  ///                                              blocking and occupancy all pay.
  ///
  /// This is why the exercise ladder ends with matmul: it is the first kernel where
  /// being clever about arithmetic is even possible.
  double arithmetic_intensity() const {
    if (bytes <= 0.0) return 0.0;
    return flops / bytes;
  }

  /// Serialize this result to a JSON object string.
  std::string to_json() const {
    std::ostringstream oss;
    oss << "    {\n";
    oss << "      \"name\": \"" << escape_json(name) << "\",\n";
    oss << "      \"size\": \"" << escape_json(size) << "\",\n";
    oss << "      \"implemented\": " << (implemented ? "true" : "false");
    if (!implemented) {
      oss << ",\n      \"note\": \"" << escape_json(note) << "\"\n";
      oss << "    }";
      return oss.str();
    }
    oss << ",\n";
    oss << "      \"median_ms\": " << stats.median_ms << ",\n";
    oss << "      \"min_ms\": " << stats.min_ms << ",\n";
    oss << "      \"max_ms\": " << stats.max_ms << ",\n";
    oss << "      \"reps\": " << stats.reps << ",\n";
    oss << "      \"spread_pct\": " << stats.spread_pct() << ",\n";
    oss << "      \"flops\": " << std::fixed << flops << ",\n";
    oss << "      \"bytes\": " << std::fixed << bytes << ",\n";
    oss << "      \"gflops\": " << gflops() << ",\n";
    oss << "      \"gbps\": " << gbps() << ",\n";
    const double peak_bw = engine::cuda_peak_bandwidth_gbs();
    const double pct_bw = (bytes > 0.0 && peak_bw > 0.0) ? (100.0 * gbps() / peak_bw) : 0.0;
    oss << "      \"pct_peak_bw\": " << pct_bw << ",\n";
    oss << "      \"arithmetic_intensity\": " << arithmetic_intensity() << "\n";
    oss << "    }";
    return oss.str();
  }
};

//===----------------------------------------------------------------------===//
// A markdown table of results.
//
// Markdown because the destination is your report and your lab notebook, and pasting
// a table beats retyping numbers -- retyping is how transcription errors get into a
// dissertation. It also renders directly on GitHub.
//===----------------------------------------------------------------------===//
class Table {
 public:
  explicit Table(std::string title) : title_(std::move(title)) {}

  void add(Result r) { rows_.push_back(std::move(r)); }

  /// Runs `fn` under time_gpu() and records the result -- but if the kernel is still
  /// a stub, records a "not implemented" row and carries on.
  ///
  /// This is what makes the benchmark binary useful from day one instead of after
  /// exercise 6. Running it prints the whole table with holes in it, and the holes
  /// fill in as you work. A harness that refuses to run until everything is finished
  /// is a harness you will not run.
#if ENGINE_HAS_CUDA
  template <typename F>
  void measure_gpu(const std::string& name, const std::string& size, double flops,
                   double bytes, F&& fn, int warmup = 5, int reps = 50) {
    Result r;
    r.name = name;
    r.size = size;
    r.flops = flops;
    r.bytes = bytes;
    try {
      r.stats = time_gpu(std::forward<F>(fn), warmup, reps);
    } catch (const std::exception& e) {
      r.implemented = false;
      r.note = short_reason(e.what());
      // Leave the device in a clean state: a throw from inside the timed loop may
      // have left a sticky CUDA error, and every later cudaXxx call would then fail
      // with the same message and make the whole table look broken.
      cudaGetLastError();
    }
    rows_.push_back(std::move(r));
  }
#endif

  template <typename F>
  void measure_cpu(const std::string& name, const std::string& size, double flops,
                   double bytes, F&& fn, int warmup = 2, int reps = 10) {
    Result r;
    r.name = name;
    r.size = size;
    r.flops = flops;
    r.bytes = bytes;
    try {
      r.stats = time_cpu(std::forward<F>(fn), warmup, reps);
    } catch (const std::exception& e) {
      r.implemented = false;
      r.note = short_reason(e.what());
    }
    rows_.push_back(std::move(r));
  }

  void print(std::ostream& os = std::cout) const {
    os << "\n### " << title_ << "\n\n";
    os << "| kernel | size | median (ms) | min (ms) | spread | GFLOP/s | GB/s | "
          "% peak BW | AI (FLOP/B) |\n";
    os << "|---|---|---:|---:|---:|---:|---:|---:|---:|\n";

    const double peak_bw = engine::cuda_peak_bandwidth_gbs();
    bool any_noisy = false;

    for (const Result& r : rows_) {
      os << "| `" << r.name << "` | " << r.size << " | ";
      if (!r.implemented) {
        // Seven cells of em-dash, then the reason. A visible gap in the table is a
        // to-do list; a missing row is something you forget you were going to do.
        os << "-- | -- | -- | -- | -- | -- | -- |";
        os << "  <!-- " << r.note << " -->\n";
        continue;
      }

      const double spread = r.stats.spread_pct();
      const bool noisy = spread > kNoisySpreadPct;
      if (noisy) any_noisy = true;

      os << num(r.stats.median_ms) << " | " << num(r.stats.min_ms) << " | "
         << num(spread) << "%" << (noisy ? " (!)" : "") << " | ";
      os << (r.flops > 0.0 ? num(r.gflops()) : std::string("--")) << " | ";
      os << (r.bytes > 0.0 ? num(r.gbps()) : std::string("--")) << " | ";
      if (r.bytes > 0.0 && peak_bw > 0.0) {
        os << num(100.0 * r.gbps() / peak_bw) << "% | ";
      } else {
        os << "-- | ";
      }
      os << (r.bytes > 0.0 ? num(r.arithmetic_intensity()) : std::string("--"))
         << " |\n";
    }

    // Provenance. A benchmark table without the hardware, the build type and the
    // date is not reproducible, and in six months you will not be able to reconstruct
    // any of it. This footer is the difference between a result and a rumour.
    os << "\nHardware: " << engine::cuda_device_summary() << "\n";
#if ENGINE_HAS_CUDA
    os << "Driver version: " << engine::cuda_driver_version() << "\n";
    const int live_sm_clock = engine::cuda_live_sm_clock_mhz();
    if (live_sm_clock > 0) {
      os << "GPU SM clock: " << live_sm_clock << " MHz (live via NVML; nominal "
         << (engine::cuda_clock_rate_khz() / 1000) << " MHz)\n";
    } else if (engine::cuda_clock_rate_khz() > 0) {
      os << "GPU clock rate: " << (engine::cuda_clock_rate_khz() / 1000) << " MHz\n";
    }
#endif
    if (peak_bw > 0.0) {
      os << "Peak DRAM bandwidth (theoretical): " << num(peak_bw) << " GB/s\n";
    }
    os << "Build: " << build_description() << "\n";

#if ENGINE_HAS_CUDA
    os << "\n_Lock the GPU clocks before quoting these numbers -- see the note at the\n"
          "top of bench/bench_harness.hpp. Record the locked clock alongside the table\n"
          "in docs/lab-notebook.md._\n";
#endif

    if (any_noisy) {
      // Said loudly, because a noisy measurement that gets copied into a report is
      // indistinguishable from a real result once it is there.
      os << "\n**(!) marks rows where (max - min) / median exceeded "
         << static_cast<int>(kNoisySpreadPct)
         << "%.** Those timings are not\nreliable enough to quote. Something else was"
            " using the machine, the clocks\nwere moving, or the kernel is too short"
            " to measure -- find out which and\nre-run before recording the number.\n";
    }
  }

  /// Print all results as a JSON document for scripting and plot generation.
  void print_json(std::ostream& os = std::cout) const {
    const double peak_bw = engine::cuda_peak_bandwidth_gbs();
    const int live_sm_clock = engine::cuda_live_sm_clock_mhz();
    const int static_sm_clock = engine::cuda_clock_rate_khz() / 1000;
    os << "{\n";
    os << "  \"title\": \"" << escape_json(title_) << "\",\n";
    os << "  \"environment\": {\n";
    os << "    \"device\": \"" << escape_json(engine::cuda_device_summary()) << "\",\n";
    os << "    \"driver_version\": \"" << escape_json(engine::cuda_driver_version()) << "\",\n";
    os << "    \"clock_rate_mhz\": " << (live_sm_clock > 0 ? live_sm_clock : static_sm_clock) << ",\n";
    os << "    \"live_sm_clock_mhz\": " << live_sm_clock << ",\n";
    os << "    \"static_sm_clock_mhz\": " << static_sm_clock << ",\n";
    os << "    \"peak_bandwidth_gbs\": " << peak_bw << ",\n";
    os << "    \"git_hash\": \"" << escape_json(ENGINE_GIT_HASH) << "\",\n";
    os << "    \"build_description\": \"" << escape_json(build_description()) << "\"\n";
    os << "  },\n";
    os << "  \"results\": [\n";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      os << rows_[i].to_json();
      if (i + 1 < rows_.size()) {
        os << ",";
      }
      os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
  }

  /// True if any row failed because the kernel is not written yet. main() uses this
  /// to print a reminder rather than to fail: an incomplete benchmark run is the
  /// expected state for most of this project.
  bool has_unimplemented() const {
    for (const Result& r : rows_) {
      if (!r.implemented) return true;
    }
    return false;
  }

 private:
  /// Rows above this spread are flagged rather than silently printed. 10% is the
  /// threshold from the header note: below it the measurement is usable, above it
  /// something other than the kernel is being measured.
  static constexpr double kNoisySpreadPct = 10.0;

  static std::string fixed(double v, int places) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(places) << v;
    return oss.str();
  }

  /// Roughly three significant figures, with the decimal places chosen by magnitude.
  ///
  /// A single fixed precision cannot serve this table: the values span from 0.004
  /// (launch overhead, in ms) to 60000 (GFLOP/s). Printing everything to one decimal
  /// place reported a compute-bound matmul's ideal-traffic bandwidth as "0.0 GB/s",
  /// which looks like a bug in the harness rather than the correct answer that a
  /// kernel doing 2*N^3 arithmetic against 3*N^2 bytes really does move almost no
  /// memory per unit time. "0.03" is information; "0.0" is a rounding artefact that
  /// invites an hour of debugging.
  static std::string num(double v) {
    const double a = std::abs(v);
    int places;
    if (a == 0.0) {
      places = 1;
    } else if (a < 0.01) {
      places = 4;
    } else if (a < 1.0) {
      places = 3;
    } else if (a < 100.0) {
      places = 2;
    } else {
      places = 1;
    }
    return fixed(v, places);
  }

  /// ENGINE_CHECK messages are multi-line (file, line, expression, then the text).
  /// A table cell needs the last line only.
  static std::string short_reason(const std::string& what) {
    const std::size_t nl = what.rfind('\n');
    std::string s = (nl == std::string::npos) ? what : what.substr(nl + 1);
    // Trim the two-space indent ENGINE_CHECK adds.
    const std::size_t first = s.find_first_not_of(" \t");
    return first == std::string::npos ? what : s.substr(first);
  }

  static std::string build_description() {
    std::ostringstream oss;
#if defined(NDEBUG)
    oss << "optimised (NDEBUG set)";
#else
    oss << "DEBUG -- assertions and per-launch synchronisation are ON, so these "
           "numbers are NOT representative";
#endif
#if ENGINE_HAS_CUDA
    oss << ", CUDA arch " << ENGINE_CUDA_ARCH_STRING;
#else
    oss << ", CPU-only build";
#endif
    oss << ", git " << ENGINE_GIT_HASH;
    oss << ", " << __DATE__ << " " << __TIME__;
    return oss.str();
  }

  std::string title_;
  std::vector<Result> rows_;
};

}  // namespace engbench
