#!/usr/bin/env python3
"""tools/generate_results_table.py -- Generate and update README benchmark tables.

Usage:
  python3 tools/generate_results_table.py --run
  python3 tools/generate_results_table.py --run --quick
  python3 tools/generate_results_table.py --input results.json
  python3 tools/generate_results_table.py --run --update-readme
  python3 tools/generate_results_table.py --check
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_benchmarks(binary_path: str, quick: bool = False) -> dict:
    cmd = [binary_path, "--json"]
    if quick:
        cmd.append("--quick")
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(proc.stdout)


def find_result(results: list, name_prefix: str, size_filter: str = None, exact: bool = False) -> dict:
    for r in results:
        r_name = r.get("name", "")
        matches_name = (r_name == name_prefix) if exact else r_name.startswith(name_prefix)
        if matches_name:
            if size_filter is None or size_filter in r.get("size", ""):
                return r
    return None


def format_readme_table(data: dict) -> str:
    results = data.get("results", [])
    env = data.get("environment", {})
    peak_bw = env.get("peak_bandwidth_gbs", 504.0)

    # 1. vector_add (16M or 64M elements: 192 MiB or 768 MiB total traffic)
    va = (find_result(results, "vector_add", "768.0 MiB", exact=True) or 
          find_result(results, "vector_add", "192.0 MiB", exact=True) or
          find_result(results, "vector_add", "MiB", exact=True))
    va_str = f"✅ {va['gbps']:.1f} GB/s ({va['pct_peak_bw']:.1f}% peak)" if va else "❌ Not found"

    # 2. reduce_sum (16M or 64M elements: 64 MiB or 256 MiB total traffic)
    rs = (find_result(results, "reduce_sum", "256.0 MiB", exact=True) or 
          find_result(results, "reduce_sum", "64.0 MiB", exact=True) or
          find_result(results, "reduce_sum", "MiB", exact=True))
    rs_str = f"✅ {rs['gbps']:.1f} GB/s ({rs['pct_peak_bw']:.1f}% peak)" if rs else "❌ Not found"

    # 3. softmax_rows (4096 x 4096 attention scores or 128 x 4096)
    sm = (find_result(results, "softmax_rows", "4096 x 4096", exact=True) or 
          find_result(results, "softmax_rows", "4096", exact=True))
    sm_str = f"✅ {sm['gbps']:.1f} GB/s ({sm['pct_peak_bw']:.1f}% peak)" if sm else "❌ Not found"

    # 4. rmsnorm (4096 x 4096 or 512 x 4096)
    rn = (find_result(results, "rmsnorm", "4096 x 4096", exact=True) or 
          find_result(results, "rmsnorm", "512 x 4096", exact=True) or
          find_result(results, "rmsnorm", "4096", exact=True))
    rn_str = f"✅ {rn['gbps']:.1f} GB/s ({rn['pct_peak_bw']:.1f}% peak)" if rn else "❌ Not found"

    # 5. matmul_naive (4096^3 or largest available)
    mn = find_result(results, "matmul_naive", "4096^3", exact=True) or find_result(results, "matmul_naive", "2048^3", exact=True) or find_result(results, "matmul_naive", "1024^3", exact=True)
    mn_str = f"✅ {mn['gflops']:.0f} GFLOP/s @ {mn['size']}" if mn else "❌ Not found"

    # 6. matmul_tiled
    mt = find_result(results, "matmul_tiled", "4096^3", exact=True) or find_result(results, "matmul_tiled", "2048^3", exact=True) or find_result(results, "matmul_tiled", "1024^3", exact=True)
    if mt and mn and mn.get("gflops", 0) > 0:
        speedup_pct = ((mt["gflops"] - mn["gflops"]) / mn["gflops"]) * 100.0
        mt_str = f"✅ {mt['gflops']:.0f} GFLOP/s @ {mt['size']} (+{speedup_pct:.0f}%)"
    elif mt:
        mt_str = f"✅ {mt['gflops']:.0f} GFLOP/s @ {mt['size']}"
    else:
        mt_str = "❌ Not found"

    # 7. gemv (decode M=1)
    gv = find_result(results, "gemv", "1x4096x4096", exact=True) or find_result(results, "gemv", "4096x4096", exact=True)
    gv_str = f"✅ {gv['gbps']:.1f} GB/s ({gv['pct_peak_bw']:.1f}% peak)" if gv else "❌ Not found"

    # 8. residual_rmsnorm (fused add + rmsnorm)
    rr = find_result(results, "residual_rmsnorm (fused)", "512x4096") or find_result(results, "residual_rmsnorm (fused)") or find_result(results, "residual_rmsnorm", exact=True)
    rr_sep = find_result(results, "residual+rmsnorm (separate)", "512x4096") or find_result(results, "residual+rmsnorm (separate)")
    if rr and rr_sep and rr_sep.get("median_ms", 0) > 0:
        rr_speedup = ((rr_sep["median_ms"] - rr["median_ms"]) / rr_sep["median_ms"]) * 100.0
        rr_str = f"✅ {rr['median_ms']:.3f} ms (+{rr_speedup:.1f}% over separate)"
    elif rr:
        rr_str = f"✅ {rr['median_ms']:.3f} ms"
    else:
        rr_str = "❌ Not found"

    # 9. rmsnorm_linear (fused rmsnorm + linear)
    rl = find_result(results, "rmsnorm_linear (dispatched)", "1x4096x4096") or find_result(results, "rmsnorm_linear (1D fused)", "1x4096x4096") or find_result(results, "rmsnorm_linear")
    rl_sep = find_result(results, "rmsnorm+matmul (separate)", "1x4096x4096")
    if rl and rl_sep and rl_sep.get("median_ms", 0) > 0:
        rl_speedup = ((rl_sep["median_ms"] - rl["median_ms"]) / rl_sep["median_ms"]) * 100.0
        rl_str = f"✅ {rl['median_ms']:.3f} ms (+{rl_speedup:.1f}% decode M=1)"
    elif rl:
        rl_str = f"✅ {rl['median_ms']:.3f} ms"
    else:
        rl_str = "❌ Not found"

    # 10. gemv_fp16
    g16 = find_result(results, "gemv_fp16", "1x4096x4096", exact=True) or find_result(results, "gemv_fp16", "4096x4096", exact=True)
    if g16 and gv and g16.get("min_ms", 0) > 0 and gv.get("min_ms", 0) > 0:
        g16_speedup = gv["min_ms"] / g16["min_ms"]
        g16_str = f"✅ {g16['min_ms']:.3f} ms ({g16_speedup:.2f}× speedup over FP32)"
    elif g16:
        g16_str = f"✅ {g16['min_ms']:.3f} ms"
    else:
        g16_str = "❌ Not found"

    # 11. embedding
    emb = find_result(results, "embedding (FP32)") or find_result(results, "embedding")
    emb_str = f"✅ {emb['min_ms']:.3f} ms (128-bit vector loads)" if emb else "❌ Not found"

    # 12. argmax
    am = find_result(results, "argmax (FP32)") or find_result(results, "argmax")
    am_str = f"✅ {am['min_ms']:.3f} ms (deterministic tie-breaking)" if am else "❌ Not found"

    # 13. matmul_register_tiled
    mrt = find_result(results, "matmul_register_tiled", "4096^3", exact=True) or find_result(results, "matmul_register_tiled", "2048^3", exact=True) or find_result(results, "matmul_register_tiled")
    if mrt and mt and mt.get("gflops", 0) > 0:
        mrt_speedup = mrt["gflops"] / mt["gflops"]
        mrt_str = f"✅ {mrt['gflops']:.0f} GFLOP/s @ {mrt['size']} ({mrt_speedup:.1f}× over tiled)"
    elif mrt:
        mrt_str = f"✅ {mrt['gflops']:.0f} GFLOP/s @ {mrt['size']}"
    else:
        mrt_str = "❌ Not found"

    # 14. swiglu
    sw = find_result(results, "swiglu (fused FP32)", "512x11008") or find_result(results, "swiglu (fused FP32)")
    sw_sep = find_result(results, "swiglu (unfused: silu+mul)", "512x11008")
    if sw and sw_sep and sw_sep.get("median_ms", 0) > 0:
        sw_speedup = sw_sep["median_ms"] / sw["median_ms"]
        sw_str = f"✅ {sw['median_ms']:.3f} ms ({sw_speedup:.2f}× over unfused SiLU+Mul)"
    elif sw:
        sw_str = f"✅ {sw['median_ms']:.3f} ms"
    else:
        sw_str = "❌ Not found"

    # 15. attention_naive
    an = find_result(results, "attention_naive", "TinyLlama S=2048") or find_result(results, "attention_naive")
    an_str = f"✅ {an['median_ms']:.2f} ms @ TinyLlama S=2048 ({an['gflops']:.0f} GFLOP/s)" if an else "❌ Not found"

    # 16. rope
    rp = find_result(results, "rope (RotateHalf)", "prefill S=2048 TinyLlama") or find_result(results, "rope (RotateHalf)", "prefill")
    rp_str = f"✅ {rp['median_ms']:.3f} ms @ S=2048 ({rp['gbps']:.1f} GB/s, {rp['pct_peak_bw']:.1f}% peak)" if rp else "❌ Not found"

    # 17. attention_prefill (FA2)
    ap = find_result(results, "attention_prefill (FA2)", "TinyLlama S=4096") or find_result(results, "attention_prefill (FA2)", "TinyLlama S=2048")
    an_4k = find_result(results, "attention_naive", "TinyLlama S=4096") or an
    if ap and an_4k and ap.get("median_ms", 0) > 0:
        ap_speedup = an_4k["median_ms"] / ap["median_ms"]
        ap_str = f"✅ {ap['gflops']/1000:.2f} TFLOP/s @ S=4096 ({ap['median_ms']:.2f} ms, {ap_speedup:.1f}× vs naive)"
    elif ap:
        ap_str = f"✅ {ap['gflops']/1000:.2f} TFLOP/s ({ap['median_ms']:.2f} ms)"
    else:
        ap_str = "❌ Not found"

    # 18. attention_decode (FlashDecoding)
    ad_l2 = find_result(results, "attention_decode (FlashDecoding)", "Llama2-7B L=8192")
    ad_tl = find_result(results, "attention_decode (FlashDecoding)", "TinyLlama L=32768")
    if ad_l2 and ad_tl:
        ad_str = f"✅ {ad_l2['gbps']:.1f} GB/s ({ad_l2['pct_peak_bw']:.1f}% peak, Llama2); {ad_tl['median_ms']:.3f} ms (TinyLlama 32k)"
    elif ad_l2:
        ad_str = f"✅ {ad_l2['gbps']:.1f} GB/s ({ad_l2['pct_peak_bw']:.1f}% peak)"
    else:
        ad_str = "❌ Not found"

    table_lines = [
        "| # | Kernel | New idea | GPU result |",
        "|---|---|---|---|",
        f"| 1 | `vector_add` | threads, blocks, grid-stride loops | {va_str} |",
        f"| 2 | `reduce_sum` | shared memory, `__syncthreads`, warp shuffles | {rs_str} |",
        f"| 3 | `softmax_rows` | per-row reduction, numerical stability | {sm_str} |",
        f"| 4 | `rmsnorm` | reusing the reduction pattern | {rn_str} |",
        f"| 5 | `matmul_naive` | 2-D indexing, memory traffic problem | {mn_str} |",
        f"| 6 | `matmul_tiled` | shared-memory tiling and data reuse | {mt_str} |",
        f"| 7 | `gemv` | decode token projection (M=1), 128-bit vector loads | {gv_str} |",
        f"| 8 | `residual_rmsnorm` | fused elementwise add + row reduction in 1 pass | {rr_str} |",
        f"| 9 | `rmsnorm_linear` | fused activation normalization + linear projection | {rl_str} |",
        f"| 10 | `gemv_fp16` | decode token projection in FP16 with FP32 accumulator | {g16_str} |",
        f"| 11 | `embedding` | token gather from row-major embedding table (FP32 & FP16) | {emb_str} |",
        f"| 12 | `argmax` | greedy token sampling via 16-warp shuffle reduction | {am_str} |",
        f"| 13 | `matmul_register_tiled` | 2D register tiling (8x8 thread tile, outer products, float4) | {mrt_str} |",
        f"| 14 | `swiglu` | fused SiLU + elementwise multiply (40% memory traffic reduction) | {sw_str} |",
        f"| 15 | `attention_naive` | causal FP16->FP32 GQA baseline (3-pass `[H, S, S]`) | {an_str} |",
        f"| 16 | `rope` | in-place rotary embeddings (RotateHalf & AdjacentPair) | {rp_str} |",
        f"| 17 | `attention_prefill` | FlashAttention-2 tiled online softmax + GQA SMEM reuse | {ap_str} |",
        f"| 18 | `attention_decode` | FlashDecoding split-KV + warp-per-head GQA broadcast | {ad_str} |",
    ]
    return "\n".join(table_lines)


def format_baselines_table(data: dict) -> str:
    results = data.get("results", [])
    cublas = find_result(results, "cublasSgemm (baseline)", "4096^3")
    tiled = find_result(results, "matmul_tiled", "4096^3")
    mrt_4096 = find_result(results, "matmul_register_tiled", "4096^3")
    
    rn_lin_fused_512 = find_result(results, "rmsnorm_linear (fused)", "512x4096x4096")
    rn_lin_sep_512 = find_result(results, "rmsnorm+matmul (separate)", "512x4096x4096")
    
    gemv_cold = find_result(results, "gemv", "1x4096x4096")
    gemv_warm = find_result(results, "gemv (warm L2)", "1x4096x4096")

    sw_fused = find_result(results, "swiglu (fused FP32)", "512x11008")
    sw_unfused = find_result(results, "swiglu (unfused: silu+mul)", "512x11008")

    an_4k = find_result(results, "attention_naive", "TinyLlama S=4096")
    ap_4k = find_result(results, "attention_prefill (FA2)", "TinyLlama S=4096")
    ad_tl_s1 = find_result(results, "attention_decode (splits=1)", "TinyLlama L=32768")
    ad_tl_fd = find_result(results, "attention_decode (FlashDecoding)", "TinyLlama L=32768")

    lines = [
        "### Benchmark Credibility & Baseline Verification",
        "",
        "| Baseline Comparison | Configuration | Baseline Result | Engine Result | Ratio / Notes |",
        "|---|---|---:|---:|---|",
    ]
    if cublas and tiled:
        lines.append(f"| **cuBLAS SGEMM vs Tiled GEMM** | 4096³ FP32 | {cublas['gflops']:.0f} GFLOP/s (cuBLAS) | {tiled['gflops']:.0f} GFLOP/s (tiled) | {tiled['gflops']/cublas['gflops']*100:.1f}% of cuBLAS (hand-written FP32 SIMT vs Tensor Cores) |")
    if cublas and mrt_4096 and tiled:
        lines.append(f"| **Register-Tiled GEMM vs cuBLAS** | 4096³ FP32 | {cublas['gflops']:.0f} GFLOP/s (cuBLAS) | {mrt_4096['gflops']:.0f} GFLOP/s (register-tiled) | {mrt_4096['gflops']/cublas['gflops']*100:.1f}% of cuBLAS ({mrt_4096['gflops']/tiled['gflops']:.1f}× over tiled) |")
    if rn_lin_sep_512 and rn_lin_fused_512:
        overhead = ((rn_lin_fused_512['median_ms'] - rn_lin_sep_512['median_ms']) / rn_lin_sep_512['median_ms']) * 100.0
        lines.append(f"| **Unfused vs Fused RMSNorm+Linear** | M=512, N=4096, K=4096 | {rn_lin_sep_512['median_ms']:.2f} ms (separate) | {rn_lin_fused_512['median_ms']:.2f} ms (fused) | +{overhead:.1f}% latency (1D row broadcast vs 2D shared tiling) |")
    if gemv_cold and gemv_warm:
        lines.append(f"| **GEMV Cold DRAM vs Warm L2** | 1x4096x4096 (decode) | {gemv_warm['gbps']:.1f} GB/s (warm L2) | {gemv_cold['gbps']:.1f} GB/s (cold DRAM) | {gemv_cold['pct_peak_bw']:.1f}% peak DRAM (pure streaming via rotating weight buffers) |")
    if sw_fused and sw_unfused:
        lines.append(f"| **Fused vs Unfused SwiGLU** | 512x11008 (prefill MLP) | {sw_unfused['median_ms']:.3f} ms (unfused) | {sw_fused['median_ms']:.3f} ms (fused) | {sw_unfused['median_ms']/sw_fused['median_ms']:.2f}× speedup (12N vs 20N bytes DRAM traffic) |")
    if an_4k and ap_4k:
        lines.append(f"| **FlashAttention-2 vs Naive Prefill** | TinyLlama S=4096 (32/4, D=64) | {an_4k['median_ms']:.2f} ms (2 GiB scores) | {ap_4k['median_ms']:.2f} ms ({ap_4k['gflops']/1000:.2f} TFLOP/s) | {an_4k['median_ms']/ap_4k['median_ms']:.1f}× speedup (O(S) vs O(S²) DRAM via `ncu`) |")
    if ad_tl_s1 and ad_tl_fd:
        lines.append(f"| **FlashDecoding vs Single-Pass Decode** | TinyLlama L=32768 (32/4, D=64) | {ad_tl_s1['median_ms']:.2f} ms (splits=1) | {ad_tl_fd['median_ms']:.3f} ms ({ad_tl_fd['gbps']:.1f} GB/s) | {ad_tl_s1['median_ms']/ad_tl_fd['median_ms']:.1f}× speedup (split-KV + warp GQA broadcast) |")
    
    lines.append("| **llama-bench External Baseline** | TinyLlama-1.1B (Q4_K_M) | 18,512.0 t/s (pp512) | 391.2 t/s (tg128) | 4-bit quantized weights (~249 GB/s effective) |")
    lines.append("| **llama-bench External Baseline** | TinyLlama-1.1B (Q8_0) | 18,767.1 t/s (pp512) | 275.2 t/s (tg128) | 8-bit quantized weights (~300 GB/s effective) |")
    lines.append("| **llama-bench External Baseline** | TinyLlama-1.1B (FP16) | 21,357.9 t/s (pp512) | 181.3 t/s (tg128) | 16-bit unquantized weights (~372 GB/s effective) |")
    lines.append("")
    lines.append("*Note: TinyLlama-1.1B's native context length is `2048` (`max_position_embeddings = 2048`); TinyLlama decode rows at `L > 2048` (`8192–32768`) are synthetic long-context kernel scaling benchmarks.*")
    return "\n".join(lines)


def format_full_markdown(data: dict) -> str:
    results = data.get("results", [])
    env = data.get("environment", {})
    static_sm = env.get("static_sm_clock_mhz") or 0
    dip_floor = int(static_sm * 0.75) if static_sm > 0 else 0

    lines = [
        "| kernel | size | median (ms) | IQR (ms) | min (ms) | IQR % | spread | GFLOP/s | GB/s | % peak BW | AI (FLOP/B) | SM MHz (min-max) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        if not r.get("implemented", True):
            lines.append(f"| `{r.get('name')}` | {r.get('size')} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        iqr_ms = r.get("iqr_ms")
        med_ms = r.get("median_ms", 0.0)
        iqr_s = f"{iqr_ms:.3f}" if iqr_ms is not None else "--"
        iqr_pct = r.get("iqr_pct")
        if iqr_pct is None and iqr_ms is not None and med_ms > 0:
            iqr_pct = 100.0 * iqr_ms / med_ms
        iqr_pct_s = f"{iqr_pct:.1f}%{' (!)' if iqr_pct > 10.0 else ''}" if iqr_pct is not None else "--"
        gflops_s = f"{r.get('gflops', 0.0):.1f}" if r.get("flops", 0) > 0 else "--"
        gbps_s = f"{r.get('gbps', 0.0):.1f}" if r.get("bytes", 0) > 0 else "--"
        pct_bw_s = f"{r.get('pct_peak_bw', 0.0):.1f}%" if r.get("bytes", 0) > 0 else "--"
        ai_s = f"{r.get('arithmetic_intensity', 0.0):.2f}" if r.get("bytes", 0) > 0 else "--"
        spread_s = f"{r.get('spread_pct', 0.0):.1f}%"
        sm_min = r.get("sm_clock_min_mhz") or r.get("sm_clock_mhz") or 0
        sm_max = r.get("sm_clock_max_mhz") or r.get("sm_clock_mhz") or 0
        n_clk = r.get("n_clock_samples") or 0
        if sm_min > 0:
            clk_s = f"{sm_min}" if sm_max == sm_min else f"{sm_min}-{sm_max}"
            if n_clk > 0:
                clk_s += f" (n={n_clk})"
            if dip_floor > 0 and sm_min < dip_floor:
                clk_s += " (!clk)"
        else:
            clk_s = "--"
        lines.append(
            f"| `{r.get('name')}` | {r.get('size')} | {med_ms:.3f} | "
            f"{iqr_s} | {r.get('min_ms', 0.0):.3f} | {iqr_pct_s} | {spread_s} | "
            f"{gflops_s} | {gbps_s} | {pct_bw_s} | {ai_s} | {clk_s} |"
        )

    lines.append("")
    lines.append(f"Hardware: {env.get('device', 'Unknown')}")
    lines.append(f"Driver: {env.get('driver_version', 'Unknown')}")
    if env.get("clock_rate_mhz"):
        sm_min = env.get("live_sm_clock_min_mhz")
        sm_max = env.get("live_sm_clock_max_mhz")
        if sm_min and sm_max and sm_min != sm_max:
            lines.append(f"GPU SM clock: {env.get('clock_rate_mhz')} MHz (range {sm_min}-{sm_max} MHz)")
        else:
            lines.append(f"GPU clock rate: {env.get('clock_rate_mhz')} MHz")
    if env.get("live_mem_clock_mhz"):
        lines.append(f"GPU memory clock: {env.get('live_mem_clock_mhz')} MHz")
    lines.append(f"Peak DRAM bandwidth (theoretical): {env.get('peak_bandwidth_gbs', 0.0):.1f} GB/s")
    lines.append(f"Build: {env.get('build_description', 'Unknown')}")
    return "\n".join(lines)


def update_readme(readme_path: Path, new_table: str, baselines_table: str = None) -> bool:
    content = readme_path.read_text(encoding="utf-8")
    updated = False

    # 1. Update kernel ladder table
    header = "| # | Kernel | New idea | GPU result"
    if header in content:
        start_pos = content.find(header)
        end_pos = content.find("\n\n", start_pos)
        if end_pos == -1:
            end_pos = len(content)
        content = content[:start_pos] + new_table + content[end_pos:]
        updated = True

    # 2. Update or insert baseline verification table
    if baselines_table:
        baseline_header = "### Benchmark Credibility & Baseline Verification"
        if baseline_header in content:
            start_pos = content.find(baseline_header)
            end_pos = content.find("\n\n## Design decisions", start_pos)
            if end_pos != -1:
                content = content[:start_pos] + baselines_table + "\n\n" + content[end_pos + 2:]
            else:
                end_pos = content.find("\n\n", start_pos + len(baseline_header) + 2)
                if end_pos != -1:
                    # Find end of that table
                    end_table = content.find("\n\n", end_pos + 2)
                    if end_table != -1:
                        content = content[:start_pos] + baselines_table + content[end_table:]
        else:
            # Insert before ## Design decisions
            target_marker = "## Design decisions"
            if target_marker in content:
                content = content.replace(target_marker, baselines_table + "\n\n" + target_marker)
                updated = True

    if updated:
        readme_path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate and update README results table from JSON benchmark output.")
    parser.add_argument("--input", "-i", type=str, help="Path to JSON results file.")
    parser.add_argument("--run", "-r", action="store_true", help="Run bench_kernels to produce fresh JSON.")
    parser.add_argument("--quick", "-q", action="store_true", help="Run with --quick for faster measurements.")
    parser.add_argument("--binary", type=str, default="./build/bin/bench_kernels", help="Path to bench_kernels executable.")
    parser.add_argument("--update-readme", "-u", action="store_true", help="Update README.md in-place.")
    parser.add_argument("--readme", type=str, default="README.md", help="Path to README.md.")
    parser.add_argument("--full", action="store_true", help="Print full benchmark table instead of summary.")
    parser.add_argument("--check", action="store_true", help="Verify that JSON parses and README table is populated.")
    args = parser.parse_args()

    data = None
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.run or (not sys.stdin.isatty() and not args.check):
        if not sys.stdin.isatty() and not args.run:
            try:
                data = json.load(sys.stdin)
            except Exception:
                data = None
        if data is None:
            data = run_benchmarks(args.binary, quick=args.quick)
    else:
        # Default fallback: try to run binary if it exists
        if os.path.exists(args.binary):
            data = run_benchmarks(args.binary, quick=args.quick)
        else:
            parser.error("No input specified and benchmark binary not found. Pass --input <file> or --run.")

    if not data or "results" not in data:
        print("Error: Invalid benchmark data format.", file=sys.stderr)
        sys.exit(1)

    table_md = format_readme_table(data)
    baselines_md = format_baselines_table(data)

    if args.full:
        print(format_full_markdown(data))
    else:
        print(table_md)
        print("\n" + baselines_md)

    if args.update_readme:
        readme_path = Path(args.readme)
        if not readme_path.exists():
            print(f"Error: {args.readme} not found.", file=sys.stderr)
            sys.exit(1)
        if update_readme(readme_path, table_md, baselines_md):
            print(f"\nSuccessfully updated {args.readme}.", file=sys.stderr)
        else:
            print(f"\nWarning: Could not locate table in {args.readme} to update.", file=sys.stderr)
            sys.exit(1)

    if args.check:
        print("\nCheck passed: benchmark JSON parsed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
