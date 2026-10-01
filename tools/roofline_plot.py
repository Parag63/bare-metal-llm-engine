#!/usr/bin/env python3
"""tools/roofline_plot.py -- Plot kernel arithmetic intensity vs achieved performance against roofline ceilings.

Generates a log-log roofline model visualization comparing all implemented kernels
against the hardware memory bandwidth and compute ceilings.

Usage:
  python3 tools/roofline_plot.py --run --quick
  python3 tools/roofline_plot.py --input results.json --output docs/roofline.png
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless / WSL environments
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)


def run_benchmarks(binary_path: str, quick: bool = False) -> dict:
    cmd = [binary_path, "--json"]
    if quick:
        cmd.append("--quick")
    print(f"Running benchmarks for roofline data: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(proc.stdout)


def plot_roofline(data: dict, output_path: str, peak_compute_gflops: float = 82600.0):
    env = data.get("environment", {})
    peak_bw_gbs = env.get("peak_bandwidth_gbs", 504.0)
    device_name = env.get("device", "NVIDIA GeForce RTX 4070 SUPER")

    results = data.get("results", [])
    if not results:
        print("Error: No benchmark results found in data.", file=sys.stderr)
        sys.exit(1)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Define AI domain (log scale: 0.01 to 1000 FLOP/byte)
    ai_min = 0.02
    ai_max = 800.0
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 1000)

    # Calculate theoretical ceilings
    bw_ceiling = peak_bw_gbs * ai_range
    roofline_curve = np.minimum(bw_ceiling, peak_compute_gflops)
    ridge_ai = peak_compute_gflops / peak_bw_gbs

    # Plot roofline boundary
    ax.loglog(ai_range, roofline_curve, "r-", linewidth=2.5, label=f"Roofline Ceiling (Peak BW: {peak_bw_gbs:.0f} GB/s, Peak: {peak_compute_gflops/1000:.1f} TFLOP/s)", zorder=3)
    
    # Draw bandwidth slope extension
    ax.loglog(ai_range[ai_range >= ridge_ai], bw_ceiling[ai_range >= ridge_ai], "r--", alpha=0.3, linewidth=1.5)
    # Draw compute ceiling extension
    ax.axhline(peak_compute_gflops, color="r", linestyle="--", alpha=0.3, linewidth=1.5)

    # Vertical line at ridge point (balance point)
    ax.axvline(ridge_ai, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.text(ridge_ai * 1.08, peak_compute_gflops * 0.45, f"Balance Point\n{ridge_ai:.1f} FLOP/B", 
            color="#444444", fontsize=9, verticalalignment="center", fontweight="medium")

    # Shaded regions
    ax.fill_between(ai_range[ai_range <= ridge_ai], 0.01, bw_ceiling[ai_range <= ridge_ai], color="#e8f4f8", alpha=0.4, label="Memory-bound region")
    ax.fill_between(ai_range[ai_range >= ridge_ai], 0.01, peak_compute_gflops, color="#fdf2e9", alpha=0.4, label="Compute-bound region")

    # Distinct colors and markers for kernel families
    family_styles = {
        "vector_add": {"color": "#1f77b4", "marker": "o", "label": "vector_add"},
        "reduce_sum": {"color": "#2ca02c", "marker": "s", "label": "reduce_sum"},
        "softmax_rows": {"color": "#ff7f0e", "marker": "^", "label": "softmax_rows"},
        "rmsnorm": {"color": "#9467bd", "marker": "D", "label": "rmsnorm"},
        "matmul_naive": {"color": "#d62728", "marker": "v", "label": "matmul_naive"},
        "matmul_tiled": {"color": "#2ca02c", "marker": "P", "label": "matmul_tiled"},
        "gemv": {"color": "#00a86b", "marker": "p", "label": "gemv (decode M=1)"},
        "cublas": {"color": "#7f7f7f", "marker": "*", "label": "cuBLAS Sgemm"},
        "residual_rmsnorm": {"color": "#8c564b", "marker": "X", "label": "residual_rmsnorm (fused)"},
        "rmsnorm_linear": {"color": "#e377c2", "marker": "h", "label": "rmsnorm_linear (fused)"},
        "other": {"color": "#17becf", "marker": "x", "label": "other"},
    }

    plotted_labels = set()

    # Kernel annotations to avoid crowded text
    for r in results:
        if not r.get("implemented", True):
            continue

        name = r.get("name", "")
        size = r.get("size", "")
        flops = r.get("flops", 0.0)
        bytes_val = r.get("bytes", 0.0)
        gflops = r.get("gflops", 0.0)
        ai = r.get("arithmetic_intensity", 0.0)

        # Skip zero/invalid points
        if ai <= 0.0 or gflops <= 0.0:
            continue

        # Classify kernel family
        key = "other"
        if "cublas" in name.lower():
            key = "cublas"
        elif "rmsnorm_linear" in name:
            key = "rmsnorm_linear"
        elif "residual_rmsnorm" in name:
            key = "residual_rmsnorm"
        elif "gemv" in name.lower():
            key = "gemv"
        elif "vector_add" in name:
            key = "vector_add"
        elif "reduce_sum" in name:
            key = "reduce_sum"
        elif "softmax" in name:
            key = "softmax_rows"
        elif "rmsnorm" in name:
            key = "rmsnorm"
        elif "matmul_tiled" in name:
            key = "matmul_tiled"
        elif "matmul_naive" in name:
            key = "matmul_naive"

        style = family_styles[key]
        plot_label = style["label"] if style["label"] not in plotted_labels else None
        if plot_label:
            plotted_labels.add(plot_label)

        size_pt = 70 if key != "cublas" else 110
        ax.scatter(ai, gflops, color=style["color"], marker=style["marker"], s=size_pt,
                   edgecolor="black", linewidth=0.6, zorder=5, label=plot_label)

        # Select representative points to label on the plot
        should_label = False
        label_text = name.replace(" (fused)", "").replace(" (baseline)", "")
        
        if key == "vector_add" and ("192.0 MiB" in size or "768.0 MiB" in size):
            should_label = True
            label_text = "vector_add (16M)"
        elif key == "reduce_sum" and ("64.0 MiB" in size or "256.0 MiB" in size):
            should_label = True
            label_text = "reduce_sum (16M)"
        elif key == "softmax_rows" and "4096 x 4096" in size:
            should_label = True
            label_text = "softmax (4096²)"
        elif key == "rmsnorm" and "4096 x 4096" in size:
            should_label = True
            label_text = "rmsnorm (4096²)"
        elif key == "matmul_naive" and ("4096^3" in size or "2048^3" in size):
            should_label = True
            label_text = f"matmul_naive ({size})"
        elif key == "matmul_tiled" and ("4096^3" in size or "2048^3" in size):
            should_label = True
            label_text = f"matmul_tiled ({size})"
        elif key == "cublas" and ("4096^3" in size or "2048^3" in size):
            should_label = True
            label_text = f"cuBLAS ({size})"
        elif key == "rmsnorm_linear" and "1x12288x4096" in size:
            should_label = True
            label_text = "fused rmsnorm_linear"
        elif key == "residual_rmsnorm" and "512x4096" in size:
            should_label = True
            label_text = "fused residual_rmsnorm"

        if should_label:
            offset_x = 1.15
            offset_y = 1.05
            if "naive" in name:
                offset_y = 0.75
            elif "cublas" in name:
                offset_y = 1.15
            elif "reduce" in name:
                offset_x = 0.55
                offset_y = 1.15
            ax.annotate(label_text, (ai, gflops),
                        xytext=(ai * offset_x, gflops * offset_y),
                        fontsize=8, fontweight="semibold", color="#222222",
                        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.6, shrinkA=3, shrinkB=3),
                        zorder=6)

    # Plot formatting
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.1, peak_compute_gflops * 1.5)
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte (ideal DRAM traffic)]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Achieved Performance [GFLOP/s]", fontsize=11, fontweight="bold")
    
    # Title & Subtitle
    title_str = f"Roofline Model Analysis — {device_name.split('|')[0].strip()}"
    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=12)

    # Gridlines
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Legend
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    plt.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Roofline plot saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate roofline plot from benchmark JSON.")
    parser.add_argument("--input", "-i", type=str, help="Path to input benchmark JSON file.")
    parser.add_argument("--run", "-r", action="store_true", help="Run bench_kernels to get fresh benchmark JSON.")
    parser.add_argument("--quick", "-q", action="store_true", help="Run with --quick for faster measurements.")
    parser.add_argument("--binary", type=str, default="./build/bin/bench_kernels", help="Path to bench_kernels executable.")
    parser.add_argument("--output", "-o", type=str, default="docs/roofline.png", help="Path for output image.")
    parser.add_argument("--peak-compute", type=float, default=82600.0, help="Peak FP32 GFLOP/s (default 82600 for RTX 4070S).")
    args = parser.parse_args()

    data = None
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.run or not sys.stdin.isatty():
        if not sys.stdin.isatty() and not args.run:
            try:
                data = json.load(sys.stdin)
            except Exception:
                data = None
        if data is None:
            data = run_benchmarks(args.binary, quick=args.quick)
    else:
        if os.path.exists(args.binary):
            data = run_benchmarks(args.binary, quick=args.quick)
        else:
            parser.error("No input provided and binary not found. Use --input or --run.")

    plot_roofline(data, args.output, peak_compute_gflops=args.peak_compute)


if __name__ == "__main__":
    main()
