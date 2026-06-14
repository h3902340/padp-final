#!/usr/bin/env python3
"""
AI-driven heterogeneous kernel placement optimizer.

Uses roofline model analysis and combinatorial search to find the optimal
CPU/GPU kernel assignment for each HeteroBench benchmark.
"""

import json
import itertools
import math
import os
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
PROFILE_FILE = RESULTS_DIR / "profile_data.json"
OUTPUT_FILE = RESULTS_DIR / "optimal_assignments.json"
ROOFLINE_FILE = RESULTS_DIR / "roofline_data.json"

# Hardware specs: TWCC Tesla V100-SXM2-32GB + Intel Xeon Gold 6154
HW_SPECS = {
    "cpu_peak_gflops": 72 * 2 * 3.0,  # 72 cores, 2 FMA/cycle, 3 GHz (conservative)
    "gpu_peak_gflops": 15700,  # V100 FP64 peak ~7.8 TFLOPS, FP32 ~15.7 TFLOPS
    "cpu_mem_bandwidth_gbps": 200,  # dual socket Xeon
    "gpu_mem_bandwidth_gbps": 900,  # V100 HBM2
    "pcie_bandwidth_gbps": 12,  # PCIe gen3 x16 effective
    "transfer_overhead_ms": 0.5,  # fixed kernel launch + sync overhead per device switch
}

# Estimated operational intensity (FLOPs/byte) per kernel type for roofline classification
KERNEL_INTENSITY = {
    "matrix": 128,      # compute-bound (GEMM-like)
    "conv": 64,
    "filter": 8,        # stencil/filter - moderate
    "reduction": 2,     # memory-bound
    "elementwise": 4,
    "sequential": 1,    # sequential/branch-heavy (hysteresis, knn)
    "default": 16,
}

# Kernel type classification per benchmark kernel
KERNEL_TYPES = {
    "ced": ["filter", "filter", "sequential", "elementwise", "sequential"],
    "sbf": ["filter", "filter", "elementwise"],
    "opf": ["filter", "filter", "filter", "filter", "matrix", "filter", "filter", "matrix"],
    "cnn": ["conv", "elementwise", "reduction", "elementwise", "matrix", "reduction"],
    "mlp": ["matrix", "elementwise", "reduction"],
    "oha": ["elementwise", "reduction", "matrix"],
    "dgr": ["elementwise", "sequential", "sequential"],
    "spf": ["matrix", "elementwise", "matrix", "elementwise"],
    "3mm": ["matrix", "matrix", "matrix"],
    "adi": ["elementwise", "filter"],
    "ppc": ["filter", "filter"],
}


def load_profile_data():
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE) as f:
            return json.load(f)
    return {}


def roofline_time_ms(intensity, data_bytes, device):
    """Predict execution time using roofline model."""
    if device == "cpu":
        peak = HW_SPECS["cpu_peak_gflops"] * 1e9
        bw = HW_SPECS["cpu_mem_bandwidth_gbps"] * 1e9
    else:
        peak = HW_SPECS["gpu_peak_gflops"] * 1e9
        bw = HW_SPECS["gpu_mem_bandwidth_gbps"] * 1e9

    flops = intensity * data_bytes
    t_compute = flops / peak * 1000  # ms
    t_memory = data_bytes / bw * 1000  # ms
    return max(t_compute, t_memory)


def normalize_kernel_names(kernels_dict, expected_names):
    """Match parsed kernel names to expected ordered list."""
    if not kernels_dict:
        return {}
    if len(kernels_dict) == len(expected_names):
        values = list(kernels_dict.values())
        return dict(zip(expected_names, values))

    # Fuzzy match by substring
    result = {}
    used = set()
    for exp in expected_names:
        exp_lower = exp.lower().replace("_", " ")
        for name, val in kernels_dict.items():
            if name in used:
                continue
            name_lower = name.lower().replace("_", " ")
            if exp_lower in name_lower or name_lower in exp_lower:
                result[exp] = val
                used.add(name)
                break
    # Fill remaining by order
    remaining_exp = [e for e in expected_names if e not in result]
    remaining_vals = [v for n, v in kernels_dict.items() if n not in used]
    for e, v in zip(remaining_exp, remaining_vals):
        result[e] = v
    return result


def estimate_data_bytes(bench, n_kernels, kernel_idx):
    """Rough data transfer size estimate per kernel boundary (bytes)."""
    sizes = {
        "ced": 1439 * 1919 * 8,
        "sbf": 1439 * 1919 * 8,
        "opf": 2160 * 3840 * 4,
        "cnn": 1024 * 2048 * 4,
        "mlp": 3072 * 4096 * 8,
        "oha": 8 * 1024 * 128 * 4,
        "dgr": 18000 * 196 * 4,
        "spf": 10000 * 5000 * 8,
        "3mm": 1024 * 1024 * 8,
        "adi": 1024 * 1024 * 8,
        "ppc": 1000000 * 24,
    }
    base = sizes.get(bench, 1024 * 1024 * 8)
    return base * (0.5 + 0.1 * kernel_idx)


def compute_transfer_cost_ms(bench, assignment, n_kernels):
    """Compute PCIe transfer cost when switching between CPU and GPU."""
    cost = 0.0
    prev = None
    for i in range(n_kernels):
        dev = assignment[i]
        if prev is not None and dev != prev:
            data_bytes = estimate_data_bytes(bench, n_kernels, i)
            pcie_time = data_bytes / (HW_SPECS["pcie_bandwidth_gbps"] * 1e9) * 1000
            cost += pcie_time + HW_SPECS["transfer_overhead_ms"]
        prev = dev
    return cost


def evaluate_assignment(bench, assignment, cpu_times, gpu_times, kernel_types):
    """Total predicted time for a kernel assignment (0=CPU, 1=GPU)."""
    total = 0.0
    kernel_names = list(cpu_times.keys())
    for i, name in enumerate(kernel_names):
        if assignment[i] == 0:
            total += cpu_times.get(name, float("inf"))
        else:
            total += gpu_times.get(name, float("inf"))
    total += compute_transfer_cost_ms(bench, assignment, len(kernel_names))
    return total


def optimize_assignment(bench, cpu_times, gpu_times, kernel_types):
    """Find optimal kernel assignment via exhaustive search (feasible for <=8 kernels)."""
    kernel_names = list(cpu_times.keys())
    n = len(kernel_names)
    if n == 0:
        return [], float("inf"), {}

    best_assignment = [0] * n
    best_time = float("inf")
    all_results = {}

    for bits in range(2 ** n):
        assignment = [(bits >> i) & 1 for i in range(n)]
        t = evaluate_assignment(bench, assignment, cpu_times, gpu_times, kernel_types)
        key = "".join("G" if a else "C" for a in assignment)
        all_results[key] = t
        if t < best_time:
            best_time = t
            best_assignment = assignment

    cpu_baseline = sum(cpu_times.values())
    speedup = cpu_baseline / best_time if best_time > 0 else 0

    return best_assignment, best_time, {
        "cpu_baseline_ms": cpu_baseline,
        "gpu_all_ms": sum(gpu_times.values()) + compute_transfer_cost_ms(
            bench, [1] * n, n),
        "best_hetero_ms": best_time,
        "speedup_vs_cpu": speedup,
        "all_assignments": all_results,
    }


def roofline_analysis(bench, kernel_names, kernel_types, cpu_times, gpu_times):
    """Generate roofline classification for each kernel."""
    analysis = []
    for i, name in enumerate(kernel_names):
        ktype = kernel_types[i] if i < len(kernel_types) else "default"
        intensity = KERNEL_INTENSITY.get(ktype, KERNEL_INTENSITY["default"])
        data_bytes = estimate_data_bytes(bench, len(kernel_names), i)

        cpu_roof = roofline_time_ms(intensity, data_bytes, "cpu")
        gpu_roof = roofline_time_ms(intensity, data_bytes, "gpu")
        cpu_actual = cpu_times.get(name, 0)
        gpu_actual = gpu_times.get(name, 0)

        # Classify: compute-bound if intensity > ridge point
        cpu_ridge = HW_SPECS["cpu_peak_gflops"] / HW_SPECS["cpu_mem_bandwidth_gbps"]
        gpu_ridge = HW_SPECS["gpu_peak_gflops"] / HW_SPECS["gpu_mem_bandwidth_gbps"]

        analysis.append({
            "kernel": name,
            "type": ktype,
            "operational_intensity": intensity,
            "cpu_ridge_point": cpu_ridge,
            "gpu_ridge_point": gpu_ridge,
            "cpu_bound": "compute" if intensity > cpu_ridge else "memory",
            "gpu_bound": "compute" if intensity > gpu_ridge else "memory",
            "cpu_measured_ms": cpu_actual,
            "gpu_measured_ms": gpu_actual,
            "cpu_roofline_ms": cpu_roof,
            "gpu_roofline_ms": gpu_roof,
            "recommended": "GPU" if gpu_actual < cpu_actual else "CPU",
        })
    return analysis


def gradient_correction(assignments, measured_hetero, alpha=0.3):
    """Reflexion-style gradient correction: update predictions with measured error."""
    corrected = {}
    for bench, data in assignments.items():
        predicted = data.get("metrics", {}).get("best_hetero_ms", 0)
        actual = measured_hetero.get(bench)
        if actual and predicted > 0:
            error = (actual - predicted) / predicted
            corrected[bench] = {
                "predicted_ms": predicted,
                "actual_ms": actual,
                "error_ratio": error,
                "corrected_prediction_ms": predicted * (1 + alpha * error),
            }
    return corrected


def main():
    profile_data = load_profile_data()
    if not profile_data:
        print("No profile data found. Run parse_profile.py after benchmarking.")
        # Use reference data from test run for 3mm
        profile_data = {
            "3mm": {
                "backends": {
                    "cpu": {"total_ms": 5509.88, "kernels": {
                        "Kernel 3mm 0": 2311.77, "Kernel 3mm 1": 2310.02, "Kernel 3mm 2": 888.074}},
                    "gpu_cuda": {"total_ms": 19.649, "kernels": {
                        "Kernel 3mm 0": 6.61016, "Kernel 3mm 1": 6.61898, "Kernel 3mm 2": 6.40392}},
                },
                "expected_kernels": BENCHMARK_KERNELS["3mm"] if "3mm" in dir() else [],
            }
        }

    assignments = {}
    roofline_data = {}

    try:
        from parse_profile import BENCHMARK_KERNELS
    except ImportError:
        BENCHMARK_KERNELS = {}

    for bench, data in profile_data.items():
        expected = BENCHMARK_KERNELS.get(bench, data.get("expected_kernels", []))
        backends = data.get("backends", {})

        cpu_data = backends.get("cpu", {})
        gpu_data = backends.get("gpu_omp") or backends.get("gpu_cuda", {})

        cpu_kernels = normalize_kernel_names(cpu_data.get("kernels", {}), expected)
        gpu_kernels = normalize_kernel_names(gpu_data.get("kernels", {}), expected)

        if not cpu_kernels or not gpu_kernels:
            print(f"Skipping {bench}: missing CPU or GPU profile data")
            continue

        ktypes = KERNEL_TYPES.get(bench, ["default"] * len(expected))
        assignment, best_time, metrics = optimize_assignment(
            bench, cpu_kernels, gpu_kernels, ktypes)

        kernel_assignment = {}
        for i, name in enumerate(cpu_kernels.keys()):
            kernel_assignment[name] = "GPU" if assignment[i] else "CPU"

        assignments[bench] = {
            "assignment": kernel_assignment,
            "assignment_bits": assignment,
            "metrics": metrics,
            "cpu_total_ms": cpu_data.get("total_ms"),
            "gpu_total_ms": gpu_data.get("total_ms"),
        }

        roofline_data[bench] = roofline_analysis(
            bench, list(cpu_kernels.keys()), ktypes, cpu_kernels, gpu_kernels)

        print(f"\n{bench}:")
        print(f"  CPU baseline: {metrics['cpu_baseline_ms']:.2f} ms")
        print(f"  Optimal hetero: {best_time:.2f} ms (speedup {metrics['speedup_vs_cpu']:.2f}x)")
        print(f"  Assignment: {kernel_assignment}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(assignments, f, indent=2)
    with open(ROOFLINE_FILE, "w") as f:
        json.dump(roofline_data, f, indent=2)

    print(f"\nSaved assignments to {OUTPUT_FILE}")
    print(f"Saved roofline analysis to {ROOFLINE_FILE}")
    return assignments, roofline_data


if __name__ == "__main__":
    main()
