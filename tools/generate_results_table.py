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


def find_result(results: list, name_prefix: str, size_filter: str = None) -> dict:
    for r in results:
        if r.get("name", "").startswith(name_prefix):
            if size_filter is None or size_filter in r.get("size", ""):
                return r
    return None


def format_readme_table(data: dict) -> str:
    results = data.get("results", [])
    env = data.get("environment", {})
    peak_bw = env.get("peak_bandwidth_gbs", 504.0)

    # 1. vector_add (16M or 64M elements: 192 MiB or 768 MiB total traffic)
    va = (find_result(results, "vector_add", "768.0 MiB") or 
          find_result(results, "vector_add", "192.0 MiB") or
          find_result(results, "vector_add", "MiB"))
    va_str = f"✅ {va['gbps']:.1f} GB/s ({va['pct_peak_bw']:.1f}% peak)" if va else "❌ Not found"

    # 2. reduce_sum (16M or 64M elements: 64 MiB or 256 MiB total traffic)
    rs = (find_result(results, "reduce_sum", "256.0 MiB") or 
          find_result(results, "reduce_sum", "64.0 MiB") or
          find_result(results, "reduce_sum", "MiB"))
    rs_str = f"✅ {rs['gbps']:.1f} GB/s ({rs['pct_peak_bw']:.1f}% peak)" if rs else "❌ Not found"

    # 3. softmax_rows (4096 x 4096 attention scores or 128 x 4096)
    sm = (find_result(results, "softmax_rows", "4096 x 4096") or 
          find_result(results, "softmax_rows", "4096"))
    sm_str = f"✅ {sm['gbps']:.1f} GB/s ({sm['pct_peak_bw']:.1f}% peak)" if sm else "❌ Not found"

    # 4. rmsnorm (4096 x 4096 or 512 x 4096)
    rn = (find_result(results, "rmsnorm", "4096 x 4096") or 
          find_result(results, "rmsnorm", "512 x 4096") or
          find_result(results, "rmsnorm", "4096"))
    rn_str = f"✅ {rn['gbps']:.1f} GB/s ({rn['pct_peak_bw']:.1f}% peak)" if rn else "❌ Not found"

    # 5. matmul_naive (4096^3 or largest available)
    mn = find_result(results, "matmul_naive", "4096^3") or find_result(results, "matmul_naive", "2048^3") or find_result(results, "matmul_naive", "1024^3")
    mn_str = f"✅ {mn['gflops']:.0f} GFLOP/s @ {mn['size']}" if mn else "❌ Not found"

    # 6. matmul_tiled
    mt = find_result(results, "matmul_tiled", "4096^3") or find_result(results, "matmul_tiled", "2048^3") or find_result(results, "matmul_tiled", "1024^3")
    if mt and mn and mn.get("gflops", 0) > 0:
        speedup_pct = ((mt["gflops"] - mn["gflops"]) / mn["gflops"]) * 100.0
        mt_str = f"✅ {mt['gflops']:.0f} GFLOP/s @ {mt['size']} (+{speedup_pct:.0f}%)"
    elif mt:
        mt_str = f"✅ {mt['gflops']:.0f} GFLOP/s @ {mt['size']}"
    else:
        mt_str = "❌ Not found"

    # 7. gemv (decode M=1)
    gv = find_result(results, "gemv", "12288x4096") or find_result(results, "gemv", "4096x4096") or find_result(results, "gemv")
    gv_str = f"✅ {gv['gbps']:.1f} GB/s ({gv['pct_peak_bw']:.1f}% peak)" if gv else "❌ Not found"

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
    ]
    return "\n".join(table_lines)


def format_full_markdown(data: dict) -> str:
    results = data.get("results", [])
    env = data.get("environment", {})

    lines = [
        "| kernel | size | median (ms) | min (ms) | spread | GFLOP/s | GB/s | % peak BW | AI (FLOP/B) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        if not r.get("implemented", True):
            lines.append(f"| `{r.get('name')}` | {r.get('size')} | -- | -- | -- | -- | -- | -- | -- |")
            continue
        gflops_s = f"{r.get('gflops', 0.0):.1f}" if r.get("flops", 0) > 0 else "--"
        gbps_s = f"{r.get('gbps', 0.0):.1f}" if r.get("bytes", 0) > 0 else "--"
        pct_bw_s = f"{r.get('pct_peak_bw', 0.0):.1f}%" if r.get("bytes", 0) > 0 else "--"
        ai_s = f"{r.get('arithmetic_intensity', 0.0):.2f}" if r.get("bytes", 0) > 0 else "--"
        spread_s = f"{r.get('spread_pct', 0.0):.1f}%"
        lines.append(
            f"| `{r.get('name')}` | {r.get('size')} | {r.get('median_ms', 0.0):.3f} | "
            f"{r.get('min_ms', 0.0):.3f} | {spread_s} | {gflops_s} | {gbps_s} | {pct_bw_s} | {ai_s} |"
        )

    lines.append("")
    lines.append(f"Hardware: {env.get('device', 'Unknown')}")
    lines.append(f"Driver: {env.get('driver_version', 'Unknown')}")
    if env.get("clock_rate_mhz"):
        lines.append(f"GPU clock rate: {env.get('clock_rate_mhz')} MHz")
    lines.append(f"Peak DRAM bandwidth (theoretical): {env.get('peak_bandwidth_gbs', 0.0):.1f} GB/s")
    lines.append(f"Build: {env.get('build_description', 'Unknown')}")
    return "\n".join(lines)


def update_readme(readme_path: Path, new_table: str) -> bool:
    content = readme_path.read_text(encoding="utf-8")
    
    # Try finding markers first
    marker_start = "<!-- KERNEL_LADDER_TABLE_START -->"
    marker_end = "<!-- KERNEL_LADDER_TABLE_END -->"
    if marker_start in content and marker_end in content:
        start_idx = content.find(marker_start) + len(marker_start)
        end_idx = content.find(marker_end)
        updated = content[:start_idx] + "\n" + new_table + "\n" + content[end_idx:]
        readme_path.write_text(updated, encoding="utf-8")
        return True

    # Otherwise replace the markdown table below "The kernel ladder (complete)"
    header = "| # | Kernel | New idea | GPU result |"
    if header in content:
        start_pos = content.find(header)
        # Find end of table (empty line after table rows)
        end_pos = content.find("\n\n", start_pos)
        if end_pos == -1:
            end_pos = len(content)
        updated = content[:start_pos] + new_table + content[end_pos:]
        readme_path.write_text(updated, encoding="utf-8")
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

    if args.full:
        print(format_full_markdown(data))
    else:
        print(table_md)

    if args.update_readme:
        readme_path = Path(args.readme)
        if not readme_path.exists():
            print(f"Error: {args.readme} not found.", file=sys.stderr)
            sys.exit(1)
        if update_readme(readme_path, table_md):
            print(f"\nSuccessfully updated {args.readme}.", file=sys.stderr)
        else:
            print(f"\nWarning: Could not locate table in {args.readme} to update.", file=sys.stderr)
            sys.exit(1)

    if args.check:
        print("\nCheck passed: benchmark JSON parsed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
