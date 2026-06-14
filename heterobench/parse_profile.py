#!/usr/bin/env python3
"""Parse HeteroBench profiling logs and extract per-kernel timing data."""

import re
import json
import glob
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = Path(__file__).parent / "results" / "profile_data.json"

# Kernel name patterns per benchmark (order matters for transfer cost)
BENCHMARK_KERNELS = {
    "ced": ["Gaussian Filter", "Gradient", "Edge Thinning", "Double Threshold", "Hysteresis"],
    "sbf": ["Sobel filter x", "Sobel filter y", "Gradient magnitude"],
    "opf": ["Gradient XY", "Gradient Z", "Gradient Weight Y", "Gradient Weight X",
            "Outer Product", "Tensor Weight Y", "Tensor Weight X", "Flow Calc"],
    "cnn": ["Conv2D", "ReLU", "Max Pooling", "Pad Input", "Dot Add", "Softmax"],
    "mlp": ["Dot Add", "Sigmoid", "Softmax"],
    "oha": ["Transpose", "Softmax", "Matmul"],
    "dgr": ["Popcount", "Update KNN", "KNN Vote"],
    "spf": ["Dot Product", "Sigmoid", "Compute Gradient", "Update Parameter"],
    "3mm": ["Kernel 3mm 0", "Kernel 3mm 1", "Kernel 3mm 2"],
    "adi": ["Init Array", "Kernel ADI"],
    "ppc": ["Compute forces", "Move particles"],
}

# Regex patterns for kernel timing lines
KERNEL_TIME_PATTERNS = [
    re.compile(r"^(.+?) time:\s+([\d.]+)\s*ms\s*$", re.IGNORECASE),
]
TOTAL_TIME_PATTERN = re.compile(r"Single iteration time:\s+([\d.]+)\s*ms", re.IGNORECASE)

SKIP_LINES = {"single iteration", "warm up", "total"}


def parse_log_file(log_path):
    """Extract kernel timings and total time from a log file."""
    kernels = {}
    total_ms = None
    with open(log_path, "r", errors="replace") as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TOTAL_TIME_PATTERN.search(line)
        if m:
            total_ms = float(m.group(1))
            continue
        for pat in KERNEL_TIME_PATTERNS:
            m = pat.match(line)
            if m:
                name = m.group(1).strip()
                if any(s in name.lower() for s in SKIP_LINES):
                    continue
                kernels[name] = float(m.group(2))
    return {"total_ms": total_ms, "kernels": kernels}


def find_logs():
    """Find all profile log files."""
    logs = {}
    for log_path in RESULTS_DIR.glob("*_cpu.log"):
        bench = log_path.stem.replace("_cpu", "")
        logs.setdefault(bench, {})["cpu"] = str(log_path)
    for log_path in RESULTS_DIR.glob("*_gpu_omp.log"):
        bench = log_path.stem.replace("_gpu_omp", "")
        logs.setdefault(bench, {})["gpu_omp"] = str(log_path)
    for log_path in RESULTS_DIR.glob("*_gpu_cuda.log"):
        bench = log_path.stem.replace("_gpu_cuda", "")
        logs.setdefault(bench, {})["gpu_cuda"] = str(log_path)
    # Also parse HeteroBench logs directory
    hb_logs = Path(__file__).parent.parent / "HeteroBench" / "HeteroBench" / "logs"
    if hb_logs.exists():
        for log_path in hb_logs.glob("*_run_*.log"):
            parts = log_path.stem.split("_")
            if len(parts) >= 2:
                bench = parts[0]
                if "cpu" in log_path.read_text(errors="replace")[:500].lower():
                    backend = "cpu" if "OpenMP" in log_path.read_text(errors="replace")[:1000] else "gpu"
                # Parse heterobench logs by content
                content = log_path.read_text(errors="replace")
                if "C++ OpenMP" in content and "heterogeneous" not in content.lower():
                    logs.setdefault(bench, {})["cpu"] = str(log_path)
                elif "CUDA GPU" in content:
                    logs.setdefault(bench, {})["gpu_cuda"] = str(log_path)
                elif "OpenMP GPU" in content or "GPU OpenMP" in content:
                    logs.setdefault(bench, {})["gpu_omp"] = str(log_path)
    return logs


def main():
    logs = find_logs()
    profile_data = {}

    for bench, backends in sorted(logs.items()):
        profile_data[bench] = {"backends": {}, "expected_kernels": BENCHMARK_KERNELS.get(bench, [])}
        for backend, log_path in backends.items():
            if os.path.exists(log_path):
                parsed = parse_log_file(log_path)
                profile_data[bench]["backends"][backend] = parsed
                print(f"Parsed {bench}/{backend}: total={parsed['total_ms']}ms, "
                      f"kernels={list(parsed['kernels'].keys())}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(profile_data, f, indent=2)
    print(f"\nSaved profile data to {OUTPUT_FILE}")
    return profile_data


if __name__ == "__main__":
    main()
