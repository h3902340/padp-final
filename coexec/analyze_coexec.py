#!/usr/bin/env python3
"""
Analyze concurrent CPU+GPU co-execution sweeps.

The optimizer does NOT brute-force every split. Instead it derives the optimal
fraction f* from a roofline-grounded performance model using only the two
single-device calibration points (f=0 pure CPU, f=1 pure GPU):

  Static-partition makespan model (work W split by fraction f to GPU):
      T(f) = max( T_cpu(1-f), T_gpu(f) )
      T_cpu(x) = x * W / R_cpu
      T_gpu(x) = x * W / R_gpu + C_transfer(x)

  Ignoring transfer, the makespan minimum occurs where the two devices finish
  together (the balance point):
      (1-f*) / R_cpu = f* / R_gpu   =>   f* = R_gpu / (R_cpu + R_gpu)

  R_cpu, R_gpu are throughputs read off the roofline:
    - compute-bound (GEMM): R = attainable GFLOP/s (FLOP roofline)
    - memory-bound (stencil): R = attainable GB/s  (bandwidth roofline)

  Predicted best makespan:  T* = W / (R_cpu + R_gpu)
  Speedup over pure GPU  =  1 + R_cpu / R_gpu
  Speedup over pure CPU  =  1 + R_gpu / R_cpu

The empirical sweep is used only to VALIDATE that the predicted f* lands at the
measured minimum, not to find it.
"""

import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).parent


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (v) for k, v in row.items()})
    return rows


def to_float(rows, keys):
    for r in rows:
        for k in keys:
            if k in r and r[k] not in (None, ""):
                r[k] = float(r[k])
    return rows


def analyze_gemm(path):
    rows = to_float(load_csv(path), ["frac", "t_total_ms", "t_cpu_ms",
                                     "t_gpu_ms", "gflops"])
    if not rows:
        return None
    N = int(float(rows[0]["N"]))
    W_flop = 2.0 * N * N * N

    # Calibration points
    cpu_pt = min((r for r in rows if abs(r["frac"]) < 1e-6), default=None,
                 key=lambda r: r["t_total_ms"])
    gpu_pt = min((r for r in rows if abs(r["frac"] - 1.0) < 1e-6), default=None,
                 key=lambda r: r["t_total_ms"])
    if not cpu_pt or not gpu_pt:
        return None

    R_cpu = W_flop / (cpu_pt["t_total_ms"] / 1000.0) / 1e9  # GFLOP/s
    R_gpu = W_flop / (gpu_pt["t_total_ms"] / 1000.0) / 1e9

    f_star = R_gpu / (R_cpu + R_gpu)
    T_pred_ms = W_flop / ((R_cpu + R_gpu) * 1e9) * 1000.0

    # Empirical best
    best = min(rows, key=lambda r: r["t_total_ms"])

    return {
        "kernel": "gemm", "N": N, "class": "compute-bound",
        "R_cpu_gflops": R_cpu, "R_gpu_gflops": R_gpu,
        "t_cpu_only_ms": cpu_pt["t_total_ms"],
        "t_gpu_only_ms": gpu_pt["t_total_ms"],
        "f_star_predicted": f_star,
        "T_predicted_ms": T_pred_ms,
        "f_empirical_best": best["frac"],
        "T_empirical_best_ms": best["t_total_ms"],
        "speedup_vs_cpu": cpu_pt["t_total_ms"] / best["t_total_ms"],
        "speedup_vs_gpu": gpu_pt["t_total_ms"] / best["t_total_ms"],
        "speedup_vs_gpu_predicted": 1.0 + R_cpu / R_gpu,
        "sweep": [(r["frac"], r["t_total_ms"]) for r in rows],
    }


def analyze_stencil(path):
    rows = to_float(load_csv(path), ["frac", "t_total_ms", "t_cpu_ms",
                                     "t_gpu_ms", "gbps"])
    if not rows:
        return None
    H = int(float(rows[0]["H"]))
    W = int(float(rows[0]["W"]))
    W_bytes = 2.0 * H * W * 4.0  # read+write float image

    cpu_pt = min((r for r in rows if abs(r["frac"]) < 1e-6), default=None,
                 key=lambda r: r["t_total_ms"])
    gpu_pt = min((r for r in rows if abs(r["frac"] - 1.0) < 1e-6), default=None,
                 key=lambda r: r["t_total_ms"])
    if not cpu_pt or not gpu_pt:
        return None

    R_cpu = W_bytes / (cpu_pt["t_total_ms"] / 1000.0) / 1e9  # GB/s
    R_gpu = W_bytes / (gpu_pt["t_total_ms"] / 1000.0) / 1e9

    f_star = R_gpu / (R_cpu + R_gpu)
    T_pred_ms = W_bytes / ((R_cpu + R_gpu) * 1e9) * 1000.0
    best = min(rows, key=lambda r: r["t_total_ms"])

    return {
        "kernel": "stencil", "H": H, "W": W, "class": "memory-bound",
        "R_cpu_gbps": R_cpu, "R_gpu_gbps": R_gpu,
        "t_cpu_only_ms": cpu_pt["t_total_ms"],
        "t_gpu_only_ms": gpu_pt["t_total_ms"],
        "f_star_predicted": f_star,
        "T_predicted_ms": T_pred_ms,
        "f_empirical_best": best["frac"],
        "T_empirical_best_ms": best["t_total_ms"],
        "speedup_vs_cpu": cpu_pt["t_total_ms"] / best["t_total_ms"],
        "speedup_vs_gpu": gpu_pt["t_total_ms"] / best["t_total_ms"],
        "speedup_vs_gpu_predicted": 1.0 + R_cpu / R_gpu,
        "sweep": [(r["frac"], r["t_total_ms"]) for r in rows],
    }


def _find(name):
    """Look for a sweep CSV in ./results then alongside this script."""
    for cand in (HERE / "results" / name, HERE / name):
        if cand.exists():
            return cand
    return HERE / "results" / name


def main():
    out = {}
    gemm_csv = _find("gemm_sweep.csv")
    stencil_csv = _find("stencil_sweep.csv")
    if gemm_csv.exists():
        g = analyze_gemm(gemm_csv)
        if g:
            out["gemm"] = g
    if stencil_csv.exists():
        s = analyze_stencil(stencil_csv)
        if s:
            out["stencil"] = s

    (HERE / "results").mkdir(exist_ok=True)
    out_path = HERE / "results" / "coexec_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    for name, d in out.items():
        print(f"\n===== {name} ({d['class']}) =====")
        if name == "gemm":
            print(f"  R_cpu = {d['R_cpu_gflops']:.1f} GFLOP/s   "
                  f"R_gpu = {d['R_gpu_gflops']:.1f} GFLOP/s")
        else:
            print(f"  R_cpu = {d['R_cpu_gbps']:.1f} GB/s   "
                  f"R_gpu = {d['R_gpu_gbps']:.1f} GB/s")
        print(f"  Predicted f* (GPU fraction) = {d['f_star_predicted']:.3f}")
        print(f"  Empirical best f            = {d['f_empirical_best']:.3f}")
        print(f"  Pure CPU : {d['t_cpu_only_ms']:.2f} ms")
        print(f"  Pure GPU : {d['t_gpu_only_ms']:.2f} ms")
        print(f"  Co-exec  : {d['T_empirical_best_ms']:.2f} ms "
              f"(predicted {d['T_predicted_ms']:.2f} ms)")
        print(f"  Speedup vs CPU = {d['speedup_vs_cpu']:.2f}x   "
              f"vs GPU = {d['speedup_vs_gpu']:.2f}x")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
